#!/bin/bash
# Standalone camera-depth obstacle node. Independent of costmap_node and
# lidar by design -- publishes its own topics, nothing downstream reads them
# yet. Same single-quoting-layer pattern as run_viz.sh / run_costmap_rgb.sh.
source /opt/ros/humble/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
cd /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap || exit 1
export PYTHONPATH=".:${PYTHONPATH:-}"
exec python3 deploy/depth_obstacle_node.py
