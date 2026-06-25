#!/bin/bash
# End-to-end validation on CARLA UE5 native ROS2 (zero carla_ros_bridge).
# Reuses a running server on :2000.
# PHASE-1: CARLA BehaviorAgent drives a long road-following route with good
#          lane keeping.
# PHASE-2: obstacle spawned on the route -> the independent native-ROS2
#          lidar safety gate (domain 0) fires a brake override -> ego stops.
LOG=/tmp/sdc_validate2.log
: > "$LOG"
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SDC=/home/merabro/selfdrive_carla_ue5
PY311=/usr/bin/python3.11
export ROS_DOMAIN_ID=0

cleanup(){
  say cleanup
  [ -n "${BRIDGE:-}" ] && kill "$BRIDGE" 2>/dev/null
  [ -n "${NODE:-}" ] && kill "$NODE" 2>/dev/null
}
trap cleanup EXIT

(echo > /dev/tcp/127.0.0.1/2000) >/dev/null 2>&1 || { say "FAIL: CARLA :2000 not up"; exit 2; }
say "CARLA server up on :2000"

# stop any stale bridge first; SIGTERM so it cleans up (kill -9 leaves
# an orphan ego blocking the spawn point)
STALE=$(ps -ef | grep -iE "[c]arla_bridge.py|[w]orld_setup/world_setup.py|[s]dc_demo_node" | awk '{print $2}')
for p in $STALE; do kill -TERM "$p" 2>/dev/null; done
sleep 5
for p in $STALE; do kill -9 "$p" 2>/dev/null; done
sleep 2

say "starting carla_bridge (py3.11: ego+sensors, native lidar, UDP pose/ctrl)"
setsid "$PY311" "$SDC/ros2_ws/src/world_setup/world_setup/carla_bridge.py" \
  > /tmp/sdc_bridge.log 2>&1 < /dev/null &
BRIDGE=$!
for i in $(seq 1 24); do
  grep -q "carla_bridge: ego id=" /tmp/sdc_bridge.log 2>/dev/null && break
  sleep 2
done
grep -q "carla_bridge: ego id=" /tmp/sdc_bridge.log 2>/dev/null \
  || { say "FAIL: bridge did not spawn ego"; tail -5 /tmp/sdc_bridge.log|tee -a "$LOG"; exit 3; }
say "bridge up: $(grep -m1 'ego id=' /tmp/sdc_bridge.log)"

# native ROS2 lidar must be visible on domain 0
source /opt/ros/humble/setup.bash 2>/dev/null
source "$SDC/ros2_ws/install/setup.bash" 2>/dev/null
for i in $(seq 1 15); do
  ros2 topic list 2>/dev/null | grep -q "/carla/ego/ego/lidar" && break; sleep 2
done
ros2 topic list 2>/dev/null | grep -q "/carla/ego/ego/lidar" \
  || { say "FAIL: native /carla/ego/ego/lidar not present on domain 0"; exit 4; }
ros2 topic list 2>/dev/null | grep -q carla_ros_bridge && { say "FAIL: ros2 bridge detected"; exit 5; }
say "native ROS2 lidar present (domain 0), zero ros2_bridge"

say "launching sdc_demo_node (py3.10 rclpy, domain 0, native-lidar safety gate)"
( ros2 run controller sdc_demo_node ) \
  > /tmp/sdc_node.log 2>&1 &
NODE=$!
for i in $(seq 1 10); do grep -q "sdc_demo up" /tmp/sdc_node.log 2>/dev/null && break; sleep 2; done
grep -q "sdc_demo up" /tmp/sdc_node.log 2>/dev/null \
  || { say "FAIL: sdc_demo_node did not start"; tail -8 /tmp/sdc_node.log|tee -a "$LOG"; exit 6; }
say "sdc_demo_node running"

# read ego speed+pos from the bridge TELEM stream (the ticking client
# is authoritative; a passive client can't see the ego in sync mode)
egostate(){
  local L
  L=$(grep '^TELEM ' /tmp/sdc_bridge.log 2>/dev/null | tail -1)
  [ -z "$L" ] && { echo "0 0 0"; return; }
  echo "$L" | sed -E 's/.*x=([-0-9.]+) y=([-0-9.]+) spd=([-0-9.]+).*/\3 \1 \2/'
}
# PHASE-1: BehaviorAgent drives a long road-following route (good lane
# keeping). Observe for 35 s and require a substantial distance.
read S0 X0 Y0 <<<"$(egostate)"
sleep 35
read S1 X1 Y1 <<<"$(egostate)"
TRAV=$(awk -v x0="$X0" -v y0="$Y0" -v x1="$X1" -v y1="$Y1" \
  'BEGIN{print sqrt((x1-x0)^2+(y1-y0)^2)}')
say "PHASE-1: speed $S0 -> $S1, travelled ${TRAV}m in 35s"
say "PHASE-1 dest: $(grep -m1 'BehaviorAgent dest=' /tmp/sdc_bridge.log)"
say "PHASE-1 telem: $(grep '^TELEM ' /tmp/sdc_bridge.log 2>/dev/null | tail -1)"
awk -v t="$TRAV" 'BEGIN{exit !(t>40.0)}' \
  || { say "FAIL PHASE-1: agent not driving a long route"; tail -5 /tmp/sdc_bridge.log|tee -a "$LOG"; exit 7; }
say "PHASE-1 PASS: BehaviorAgent drove ${TRAV}m (lane-keeping, native ROS2 up)"

# PHASE-2: obstacle ~12 m ahead -> native lidar TTC must brake ego.
# The bridge (only client that sees the ego in sync mode) spawns it.
touch /tmp/sdc_spawn_obstacle
for i in $(seq 1 15); do
  grep -q "^OBSTACLE " /tmp/sdc_bridge.log 2>/dev/null && break; sleep 1
done
grep -q "^OBSTACLE spawned" /tmp/sdc_bridge.log 2>/dev/null \
  || { say "FAIL PHASE-2: obstacle not spawned"; tail -5 /tmp/sdc_bridge.log|tee -a "$LOG"; exit 8; }
say "obstacle spawned; observing native-lidar safety gate"
sleep 16
read S2 X2 Y2 <<<"$(egostate)"
say "PHASE-2: speed with obstacle = $S2"
say "PHASE-2 telem: $(grep '^TELEM ' /tmp/sdc_bridge.log 2>/dev/null | tail -1)"
# the native-ROS2 node must have fired its brake override (ovr=1) -- this
# is the proof ROS2 (consuming the native lidar on domain 0) is in the loop
grep -q 'ovr=1' /tmp/sdc_bridge.log 2>/dev/null \
  || { say "FAIL PHASE-2: native-ROS2 lidar gate never fired (ovr stayed 0)"; tail -8 /tmp/sdc_node.log|tee -a "$LOG"; exit 8; }
say "native-ROS2 lidar gate fired brake override (ovr=1 seen in telem)"
awk -v s="$S2" 'BEGIN{exit !(s<1.0)}' \
  || { say "FAIL PHASE-2: ego did not stop for obstacle"; tail -5 /tmp/sdc_bridge.log|tee -a "$LOG"; exit 8; }
say "PHASE-2 PASS: native-ROS2 lidar gate stopped the ego for the obstacle"

say "SUCCESS: BehaviorAgent drives + native-ROS2 lidar gate works (zero ros2_bridge)"
exit 0
