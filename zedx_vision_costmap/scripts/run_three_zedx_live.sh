#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 && $# -ne 3 ]]; then
  echo "Usage: $0 [FRONT_SERIAL LEFT_SERIAL RIGHT_SERIAL]" >&2
  exit 2
fi

front_serial=${1:-42569280}
left_serial=${2:-49910017}
right_serial=${3:-43779087}

for serial in "$front_serial" "$left_serial" "$right_serial"; do
  if [[ ! "$serial" =~ ^[1-9][0-9]*$ ]]; then
    echo "Camera serials must be positive integers: $serial" >&2
    exit 2
  fi
done

if [[ "$front_serial" == "$left_serial" ||
      "$front_serial" == "$right_serial" ||
      "$left_serial" == "$right_serial" ]]; then
  echo "Front, left, and right camera serials must be distinct." >&2
  exit 2
fi

exec ros2 launch zedx_vision_costmap three_zedx_live.launch.py \
  front_serial:="$front_serial" \
  left_serial:="$left_serial" \
  right_serial:="$right_serial" \
  rviz:=true
