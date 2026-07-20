#!/usr/bin/env bash
set -uo pipefail

failures=0
check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    echo "OK command: $1"
  else
    echo "MISSING command: $1"
    failures=$((failures + 1))
  fi
}

for command in python3 ros2 colcon nvidia-smi; do
  check_command "$command"
done

echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-unset}"

if command -v ros2 >/dev/null 2>&1; then
  for package in rclpy cv_bridge nav2_costmap_2d rviz2 zed_wrapper; do
    if ros2 pkg prefix "$package" >/dev/null 2>&1; then
      echo "OK ROS package: $package"
    else
      echo "MISSING ROS package: $package"
      failures=$((failures + 1))
    fi
  done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

if (( failures > 0 )); then
  echo "Environment check failed: $failures missing requirement(s)."
  exit 1
fi
echo "Environment check passed."
