#!/bin/bash
# Desktop-icon entry point: bring the costmap RViz to the front, launching it
# only if it isn't already running.
#
# GL: NoMachine virtual displays give no usable hardware GL context, so
# rviz2 segfaults on startup ("libGL error: failed to create drawable") when
# it picks up libGLX_nvidia. Forcing the Mesa/llvmpipe software stack is what
# makes it render at all -- it then reports a real GL 4.5 context.
#
# CONFIG: costmap_cams.rviz, not costmap_live.rviz. The latter also carries a
# RobotModel (missing zedx.stl mesh) and two extra full point clouds; under a
# software rasteriser that combination crashes within ~15 s. Trimmed to the
# costmap_rgb cloud + the 3 camera panels it stays up indefinitely.
#
# ROS_DOMAIN_ID is pinned to 0 on purpose: ~/.bashrc sets 42, and anything
# that picks that up lands on an isolated DDS domain where every topic looks
# dead. Desktop launchers don't source .bashrc, but being explicit costs
# nothing and removes the trap entirely.
if pgrep -f "rviz2 -d" >/dev/null; then
  wmctrl -a 'RViz' 2>/dev/null && exit 0
fi

export ROS_DOMAIN_ID=0
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

export __GLX_VENDOR_LIBRARY_NAME=mesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export QT_X11_NO_MITSHM=1

exec rviz2 -d /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/deploy/costmap_cams.rviz
