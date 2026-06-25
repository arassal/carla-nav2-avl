#!/bin/bash
# Rerun visualization node + viewer. Run after the bridge is up.
SDC=/home/merabro/selfdrive_carla_ue5
export ROS_DOMAIN_ID=0          # CARLA native ROS2 is on domain 0
# put the bundled rerun viewer binary on PATH for rr.init(spawn=True)
export PATH="/home/merabro/.local/lib/python3.10/site-packages/rerun_sdk/rerun_cli:$PATH"
source /opt/ros/humble/setup.bash
source "$SDC/ros2_ws/install/setup.bash"

echo "[run_rerun] starting sdc_rerun_node (spawns the Rerun viewer)"
exec ros2 run controller sdc_rerun_node
