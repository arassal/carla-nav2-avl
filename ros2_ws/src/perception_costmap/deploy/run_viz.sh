#!/bin/bash
# Single-quoting-layer runner for the viz node (tmux window command was
# clobbering PYTHONPATH through nested sh -c expansion).
source /opt/ros/humble/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
cd /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap || exit 1
export PYTHONPATH=".:${PYTHONPATH:-}"
exec python3 tools/viz_node.py
