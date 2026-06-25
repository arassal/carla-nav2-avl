#!/bin/bash
# Connectivity spike: with CARLA running + ego spawned, capture the exact
# native ROS2 interface. Writes docs/native_ros2_interface.md.
set +e
source /opt/ros/humble/setup.bash
OUT=/home/merabro/selfdrive_carla_ue5/docs/native_ros2_interface.md
{
  echo "# CARLA UE5 native ROS2 interface (captured $(date))"
  echo
  echo '## ros2 topic list'
  echo '```'
  ros2 topic list
  echo '```'
  echo
  echo '## topic -> type'
  echo '```'
  for t in $(ros2 topic list); do
    echo "$t -> $(ros2 topic type "$t" 2>/dev/null)"
  done
  echo '```'
  echo
  echo '## control topic interface'
  echo '```'
  CT=$(ros2 topic list | grep -m1 -i "vehicle_control")
  echo "control topic: $CT"
  ros2 topic type "$CT" 2>/dev/null
  ros2 interface show "$(ros2 topic type "$CT" 2>/dev/null)" 2>/dev/null
  echo '```'
  echo
  echo '## /tf sample'
  echo '```'
  timeout 5 ros2 topic echo /tf --once 2>/dev/null
  echo '```'
} | tee "$OUT"
