#!/bin/bash
# Bring up the full stack: kills any stale instances, then starts one
# of each (bridge, Nav2, nodes, Rerun). CARLA must be up on :2000.
SDC=/home/merabro/selfdrive_carla_ue5
export ROS_DOMAIN_ID=0
RR_CLI=/home/merabro/.local/lib/python3.10/site-packages/rerun_sdk/rerun_cli
SRC="source /opt/ros/humble/setup.bash; source $SDC/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=0"

(echo > /dev/tcp/127.0.0.1/2000) >/dev/null 2>&1 || { echo "CARLA :2000 DOWN - start it first (scripts/run_carla_server pkg)"; exit 1; }

echo "[stack] killing stale instances..."
ps -ef | grep -v grep | grep -E 'carla_bridge[.]py|(controller |controller/lib/controller/)(sdc_demo_node|sdc_rerun_node|carla_localization|lane_path_node|cmd_to_carla)|controller_server|nav2_lifecycle_manager|lifecycle_manager|rerun --port|rerun_cli/rerun' | grep -v 'run_stack.sh' | awk '{print $2}' | sort -u > /tmp/stack_kill.txt
while read p; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null; done < /tmp/stack_kill.txt
sleep 6
while read p; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done < /tmp/stack_kill.txt
sleep 2

bg(){ nohup setsid bash -c "$1" >"$2" 2>&1 </dev/null & disown; }

echo "[stack] bridge (Nav2 mode)"
bg "exec /usr/bin/python3.11 $SDC/ros2_ws/src/world_setup/world_setup/carla_bridge.py" /tmp/sdc_bridge.log
for i in $(seq 1 25); do grep -q 'Nav2 mode' /tmp/sdc_bridge.log 2>/dev/null && break; sleep 1; done

echo "[stack] controller_server + lifecycle_manager"
bg "$SRC; exec ros2 run nav2_controller controller_server --ros-args --params-file $SDC/scripts/nav2.yaml" /tmp/sdc_ctrl.log
bg "$SRC; exec ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -p use_sim_time:=true -p autostart:=true -p node_names:=\"['controller_server']\" -p bond_timeout:=0.0" /tmp/sdc_lcm.log
sleep 12
source /opt/ros/humble/setup.bash; source $SDC/ros2_ws/install/setup.bash
# timeout-wrap: ros2 lifecycle calls block forever if the service is
# not ready, which would hang the whole bringup
ST="$(timeout 8 ros2 lifecycle get /controller_server 2>/dev/null)"
if [ "$ST" != "active [3]" ]; then
  echo "[stack] force-activating controller_server (was: ${ST:-none})"
  timeout 10 ros2 lifecycle set /controller_server configure 2>/dev/null
  timeout 10 ros2 lifecycle set /controller_server activate 2>/dev/null
  timeout 10 ros2 lifecycle set /controller_server activate 2>/dev/null
fi
echo "[stack] controller_server: $(timeout 8 ros2 lifecycle get /controller_server 2>/dev/null)"

echo "[stack] localization / lane_path / cmd_to_carla / safety gate"
# must share CARLA sim time: the native lidar is sim-time stamped, so
# wall-clock TF/odom makes the costmap drop every cloud
ST="--ros-args -p use_sim_time:=true"
bg "$SRC; exec ros2 run controller carla_localization $ST" /tmp/sdc_loc.log
bg "$SRC; exec ros2 run controller lane_path_node $ST"     /tmp/sdc_lane_path_node.log
bg "$SRC; exec ros2 run controller cmd_to_carla $ST"       /tmp/sdc_cmd_to_carla.log
bg "$SRC; exec ros2 run controller sdc_demo_node"          /tmp/sdc_sdc_demo_node.log

echo "[stack] Rerun viz (spawns one fresh viewer)"
bg "export PATH=$RR_CLI:\$PATH; $SRC; exec ros2 run controller sdc_rerun_node" /tmp/sdc_rerun.log
sleep 10

echo "[stack] node instances (want 1 each):"
timeout 8 ros2 node list 2>/dev/null | grep -E 'sdc_demo|sdc_rerun|cmd_to_carla|lane_path|carla_localization|controller_server' | sort | uniq -c
echo "[stack] rerun viewer procs: $(ps -ef|grep -v grep|grep -c 'rerun --port')"
echo "[stack] up. Logs: /tmp/sdc_*.log"
