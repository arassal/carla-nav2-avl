#!/bin/bash
# Record the complete perception contract for deterministic offline replay.
set -eo pipefail

OUT=${1:-/home/dinosaur/bags/perception_$(date +%Y%m%d_%H%M%S)}
source /opt/ros/humble/setup.bash
source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash
set -u
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

exec ros2 bag record -o "$OUT" \
  --compression-mode file --compression-format zstd --compression-threads 2 \
  /zed_front/zed_node/rgb/color/rect/image \
  /zed_front/zed_node/rgb/color/rect/camera_info \
  /zed_front/zed_node/depth/depth_registered \
  /zed_front/zed_node/confidence/confidence_map \
  /zed_left/zed_node/rgb/color/rect/image \
  /zed_left/zed_node/rgb/color/rect/camera_info \
  /zed_left/zed_node/depth/depth_registered \
  /zed_left/zed_node/confidence/confidence_map \
  /zed_right/zed_node/rgb/color/rect/image \
  /zed_right/zed_node/rgb/color/rect/camera_info \
  /zed_right/zed_node/depth/depth_registered \
  /zed_right/zed_node/confidence/confidence_map \
  /wheel_odom /tf /tf_static \
  /perception/costmap /perception/known /perception/costmap_cloud \
  /diagnostics
