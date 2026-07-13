#!/bin/bash
# Boot-time RViz: costmap (colorized) + the 3 ZED camera panels.
#
# The NoMachine virtual X display does NOT exist at boot -- it is created when
# a client connects (verified: boot 14:02:51, /tmp/.X11-unix/X1001 created
# 14:06:36 on connect). So we cannot simply export a fixed DISPLAY at startup.
# Instead: poll until some X display exists, run RViz on it, and if that
# session goes away (client disconnects, display torn down, RViz closed), fall
# back to waiting so it comes up again on the next connect.
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

CFG=/home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/deploy/costmap_live.rviz

while true; do
  sock=""
  while [ -z "$sock" ]; do
    sock=$(ls -t /tmp/.X11-unix/X* 2>/dev/null | head -1)
    [ -z "$sock" ] && sleep 5
  done
  export DISPLAY=":${sock##*/X}"
  echo "[run_rviz] display $DISPLAY up -- starting rviz2"
  rviz2 -d "$CFG"
  echo "[run_rviz] rviz2 exited; waiting for a display again"
  sleep 5
done
