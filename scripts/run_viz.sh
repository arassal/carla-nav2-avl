#!/bin/bash
# RViz visualization node + RViz2. Run after the bridge is up.
SDC=/home/merabro/selfdrive_carla_ue5
export ROS_DOMAIN_ID=0          # CARLA native ROS2 publishes on domain 0
source /opt/ros/humble/setup.bash
source "$SDC/ros2_ws/install/setup.bash"

cleanup(){ [ -n "${VIZ:-}" ] && kill "$VIZ" 2>/dev/null; }
trap cleanup EXIT

echo "[run_viz] starting sdc_viz_node (republish lidar + path/markers)"
ros2 run controller sdc_viz_node > /tmp/sdc_viz.log 2>&1 &
VIZ=$!
sleep 3
grep -q "sdc_viz up" /tmp/sdc_viz.log 2>/dev/null \
  && echo "[run_viz] $(grep -m1 'sdc_viz up' /tmp/sdc_viz.log)" \
  || { echo "[run_viz] viz node failed:"; tail -8 /tmp/sdc_viz.log; exit 1; }

echo "[run_viz] launching RViz2 (fixed frame: map)"
rviz2 -d "$SDC/scripts/sdc.rviz"
