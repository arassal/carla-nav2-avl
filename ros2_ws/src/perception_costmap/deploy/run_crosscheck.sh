#!/bin/bash
# Phase A depth/RGB cross-check monitor. READ-ONLY -- subscribes only, never
# touches the costmap or control path. Requires depth_obstacle_node running
# (run_depth_obstacle.sh). Same single-quoting-layer pattern as run_viz.sh.
source /opt/ros/humble/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
cd /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap || exit 1
export PYTHONPATH=".:${PYTHONPATH:-}"
exec python3 deploy/depth_rgb_crosscheck.py "$@"
