#!/bin/bash
# Single-quoting-layer runner (same pattern as run_viz.sh: nested sh -c
# expansion in the tmux command was clobbering the ROS env).
source /opt/ros/humble/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
cd /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap || exit 1
export PYTHONPATH=".:${PYTHONPATH:-}"
exec python3 deploy/costmap_rgb_node.py
