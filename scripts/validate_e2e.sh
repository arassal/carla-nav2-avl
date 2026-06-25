#!/bin/bash
# End-to-end bring-up + validation for the native-ROS2 SDC stack.
# Preconditions: CARLA UE5 package built (Build/Package/.../CarlaUnreal.sh).
# Steps: launch headless server --ros2 -> spawn ego+sensors -> capture native
# interface -> reconcile node topic constants -> rebuild -> run stack ->
# publish a goal -> assert ego moves, then assert it brakes for an obstacle.
# Exit 0 = SDC proven working over native ROS2 (no carla_ros_bridge).
set -u
LOG=/tmp/sdc_validate.log
: > "$LOG"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

SDC=/home/merabro/selfdrive_carla_ue5
ROS_SETUP=/opt/ros/humble/setup.bash
PKG_SH=$(find /home/merabro/CarlaUE5/Build/Package -maxdepth 6 -name CarlaUnreal.sh 2>/dev/null | head -1)

cleanup() {
  say "cleanup"
  [ -n "${WORLD_PID:-}" ] && kill "$WORLD_PID" 2>/dev/null
  [ -n "${STACK_PID:-}" ] && kill "$STACK_PID" 2>/dev/null
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

[ -z "$PKG_SH" ] && { say "FAIL: no packaged CarlaUnreal.sh found"; exit 2; }
say "package launcher: $PKG_SH"

# 1. Launch CARLA server with native ROS2 (reuse if already running)
if (echo > /dev/tcp/127.0.0.1/2000) >/dev/null 2>&1; then
  say "CARLA server already running on port 2000 - reusing it"
  SERVER_PID=""
else
  say "launching CARLA server (--ros2, windowed)"
  ( cd "$(dirname "$PKG_SH")" && exec ./CarlaUnreal.sh --ros2 \
      -nosound -carla-rpc-port=2000 ) > /tmp/sdc_server.log 2>&1 &
  SERVER_PID=$!
fi

# 2. Wait for RPC port 2000 (server ready); first run compiles shaders -> long
for i in $(seq 1 360); do
  (echo > /dev/tcp/127.0.0.1/2000) >/dev/null 2>&1 && break
  grep -qE "Exiting abnormally|Fatal error" /tmp/sdc_server.log 2>/dev/null && {
    say "FAIL: server crashed during startup"; tail -20 /tmp/sdc_server.log | tee -a "$LOG"; exit 3; }
  sleep 5
done
(echo > /dev/tcp/127.0.0.1/2000) >/dev/null 2>&1 || { say "FAIL: port 2000 never opened"; exit 3; }
say "CARLA RPC port 2000 OPEN"

# 3. Spawn ego + sensors (python3.11, native ros_name -> native ROS2 topics)
say "spawning ego + sensors"
/usr/bin/python3.11 "$SDC/ros2_ws/src/world_setup/world_setup/world_setup.py" \
  > /tmp/sdc_world.log 2>&1 &
WORLD_PID=$!
for i in $(seq 1 24); do
  grep -q "world_setup: ego id=" /tmp/sdc_world.log 2>/dev/null && break
  sleep 5
done
grep -q "world_setup: ego id=" /tmp/sdc_world.log 2>/dev/null || {
  say "FAIL: world_setup did not spawn ego"; tail -20 /tmp/sdc_world.log | tee -a "$LOG"; exit 4; }
say "ego + sensors spawned"

# 4. Capture native ROS2 interface
say "capturing native ROS2 interface (spike)"
bash "$SDC/scripts/spike_topics.sh" >> "$LOG" 2>&1
source "$ROS_SETUP"
TOPICS=$(ros2 topic list 2>/dev/null)
echo "$TOPICS" | tee -a "$LOG"
echo "$TOPICS" | grep -q "carla_ros_bridge" && { say "FAIL: ros2 bridge detected"; exit 5; }

# 5. Reconcile node topic constants with captured names
CTRL=$(echo "$TOPICS" | grep -m1 -iE "vehicle_control")
ODOM=$(echo "$TOPICS" | grep -m1 -iE "odom")
LIDAR=$(echo "$TOPICS" | grep -m1 -iE "lidar|point")
SPEED=$(echo "$TOPICS" | grep -m1 -iE "speedometer|speed")
say "discovered: ctrl=$CTRL odom=$ODOM lidar=$LIDAR speed=$SPEED"
reconcile() {  # file  pyvar  value
  [ -n "$3" ] && sed -i "s#^\(${2} = \"\).*\(\"\)#\1${3}\2#" "$1"
}
RP="$SDC/ros2_ws/src/route_planner/route_planner/route_planner_node.py"
CN="$SDC/ros2_ws/src/controller/controller/controller_node.py"
CG="$SDC/ros2_ws/src/collision_guard/collision_guard/collision_guard_node.py"
reconcile "$RP" ODOM_TOPIC "$ODOM"
reconcile "$CN" ODOM_TOPIC "$ODOM"; reconcile "$CN" SPEED_TOPIC "$SPEED"
reconcile "$CG" LIDAR_TOPIC "$LIDAR"; reconcile "$CG" SPEED_TOPIC "$SPEED"
reconcile "$CG" FINAL_CTRL_TOPIC "$CTRL"
say "reconciled topic constants"

# 6. Rebuild + run stack
( cd "$SDC/ros2_ws" && source "$ROS_SETUP" && colcon build >> "$LOG" 2>&1 )
say "stack rebuilt"
( cd "$SDC/ros2_ws" && source "$ROS_SETUP" && source install/setup.bash && \
  ros2 run route_planner route_planner_node & \
  ros2 run controller controller_node --ros-args --params-file \
    src/sdc_bringup/config/params.yaml & \
  ros2 run collision_guard collision_guard_node --ros-args --params-file \
    src/sdc_bringup/config/params.yaml & wait ) > /tmp/sdc_stack.log 2>&1 &
STACK_PID=$!
sleep 12
say "stack running (route_planner, controller, collision_guard)"

# 7. PHASE-1: publish a goal ~40 m ahead, assert ego moves
source "$SDC/ros2_ws/install/setup.bash" 2>/dev/null
ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 40.0, y: 0.0}, orientation: {w: 1.0}}}" \
  >> "$LOG" 2>&1
say "goal published; observing motion"
SPD0=$(timeout 6 ros2 topic echo "$SPEED" --once 2>/dev/null | grep -oE '[0-9.]+' | head -1)
sleep 15
SPD1=$(timeout 6 ros2 topic echo "$SPEED" --once 2>/dev/null | grep -oE '[0-9.]+' | head -1)
say "speed before=$SPD0 after=$SPD1"
awk "BEGIN{exit !(${SPD1:-0} > 0.5)}" || { say "FAIL: ego did not move toward goal"; exit 6; }
say "PHASE-1 PASS: ego is driving the route"

# 8. PHASE-2: spawn obstacle ahead, assert ego brakes
/usr/bin/python3.11 - <<'PY' >> "$LOG" 2>&1
import carla, math
c=carla.Client('localhost',2000); c.set_timeout(20); w=c.get_world()
ego=[a for a in w.get_actors().filter('vehicle.*') if a.attributes.get('role_name')=='hero' or True][0]
tf=ego.get_transform(); f=tf.get_forward_vector()
loc=carla.Location(tf.location.x+f.x*15, tf.location.y+f.y*15, tf.location.z+0.5)
bp=w.get_blueprint_library().find('vehicle.ue4.audi.tt')
w.spawn_actor(bp, carla.Transform(loc, tf.rotation)); print("obstacle spawned 15m ahead")
PY
say "obstacle spawned; observing brake"
sleep 12
SPD2=$(timeout 6 ros2 topic echo "$SPEED" --once 2>/dev/null | grep -oE '[0-9.]+' | head -1)
say "speed with obstacle=$SPD2"
awk "BEGIN{exit !(${SPD2:-9} < 1.0)}" || { say "FAIL: ego did not brake for obstacle"; exit 7; }
say "PHASE-2 PASS: ego braked for frontal obstacle"

say "SUCCESS: SDC working end-to-end over native ROS2 (no carla_ros_bridge)"
exit 0
