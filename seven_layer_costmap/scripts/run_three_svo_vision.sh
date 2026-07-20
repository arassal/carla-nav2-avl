#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 FRONT.svo2 LEFT.svo2 RIGHT.svo2 [quality|realtime]" >&2
  exit 2
fi

front_svo=$1
left_svo=$2
right_svo=$3
profile=${4:-quality}

for path in "$front_svo" "$left_svo" "$right_svo"; do
  if [[ "$path" != /* || ! -f "$path" ]]; then
    echo "SVO paths must be absolute existing files: $path" >&2
    exit 2
  fi
done

share=$(ros2 pkg prefix --share seven_layer_costmap)
case "$profile" in
  quality)
    override="$share/config/zed_svo_override.yaml"
    ;;
  realtime)
    override="$share/config/zed_svo_realtime_override.yaml"
    ;;
  *)
    echo "Profile must be quality or realtime: $profile" >&2
    exit 2
    ;;
esac

exec ros2 launch seven_layer_costmap three_svo_costmap.launch.py \
  front_svo:="$front_svo" \
  left_svo:="$left_svo" \
  right_svo:="$right_svo" \
  zed_override_path:="$override" \
  rviz:=true
