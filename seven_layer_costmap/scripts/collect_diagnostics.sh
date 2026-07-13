#!/usr/bin/env bash
set -u

output_dir="${1:-seven_layer_diagnostics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$output_dir"

ros2 node list >"$output_dir/nodes.txt" 2>&1 || true
ros2 topic list -t >"$output_dir/topics.txt" 2>&1 || true
ros2 doctor --report >"$output_dir/ros2_doctor.txt" 2>&1 || true
timeout 5 ros2 topic echo --once /seven_layer_costmap/perception_status \
  >"$output_dir/perception_status.txt" 2>&1 || true
timeout 5 ros2 topic echo --once /seven_layer_costmap/diagnostics \
  >"$output_dir/diagnostics.txt" 2>&1 || true
timeout 10 ros2 topic hz /seven_layer_costmap/costmap \
  >"$output_dir/costmap_rate.txt" 2>&1 || true
nvidia-smi >"$output_dir/nvidia_smi.txt" 2>&1 || true
python3 --version >"$output_dir/python.txt" 2>&1 || true
{
  echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
  echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"
  echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-unset}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "ZED_SDK_ROOT_DIR=${ZED_SDK_ROOT_DIR:-unset}"
} >"$output_dir/environment.txt"

echo "Diagnostics collected in: $output_dir"
