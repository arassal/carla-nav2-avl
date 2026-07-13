#!/bin/bash
# Desktop-icon entry point: bring the costmap RViz to the front, launching it
# only if it isn't already running (the boot stack usually has it up via
# run_rviz.sh, so the common case is just a focus).
if pgrep -x rviz2 >/dev/null; then
  wmctrl -a 'RViz' 2>/dev/null && exit 0
fi
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml
exec rviz2 -d /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/deploy/costmap_live.rviz
