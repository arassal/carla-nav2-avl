#!/bin/bash
# Nav2 bringup: localization, controller_server (RPP), lane_path,
# cmd_to_carla and the safety gate. Run after carla_bridge.py is up.
# (run_stack.sh does this plus the bridge and Rerun.)
SDC=/home/merabro/selfdrive_carla_ue5
export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source "$SDC/ros2_ws/install/setup.bash"
PARAMS="$SDC/scripts/nav2.yaml"

pids=()
cleanup(){ for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done; }
trap cleanup EXIT

echo "[nav2] carla_localization"
ros2 run controller carla_localization > /tmp/sdc_loc.log 2>&1 & pids+=($!)
echo "[nav2] controller_server (RPP + lidar costmap)"
ros2 run nav2_controller controller_server \
  --ros-args --params-file "$PARAMS" > /tmp/sdc_ctrl.log 2>&1 & pids+=($!)
echo "[nav2] lifecycle_manager"
ros2 run nav2_lifecycle_manager lifecycle_manager \
  --ros-args -p use_sim_time:=false -p autostart:=true \
  -p "node_names:=['controller_server']" \
  -p bond_timeout:=0.0 > /tmp/sdc_lcm.log 2>&1 & pids+=($!)
sleep 8
# force-activate: lifecycle autostart can time out over Fast-DDS
if [ "$(ros2 lifecycle get /controller_server 2>/dev/null)" != "active [3]" ]; then
  echo "[nav2] force-activating controller_server"
  ros2 lifecycle set /controller_server configure 2>/dev/null
  ros2 lifecycle set /controller_server activate 2>/dev/null
fi
echo "[nav2] lane_path (FollowPath driver)"
ros2 run controller lane_path_node > /tmp/sdc_lane.log 2>&1 & pids+=($!)
echo "[nav2] cmd_to_carla"
ros2 run controller cmd_to_carla > /tmp/sdc_cmd.log 2>&1 & pids+=($!)
echo "[nav2] native-lidar safety gate"
ros2 run controller sdc_demo_node > /tmp/sdc_node.log 2>&1 & pids+=($!)

echo "[nav2] all up. logs in /tmp/sdc_*.log . Ctrl-C to stop."
wait
