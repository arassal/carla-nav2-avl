#!/usr/bin/env bash
set -euo pipefail

topics=(
  /zedx_vision_costmap/layers/lanelet
  /zedx_vision_costmap/layers/static_obstacle
  /zedx_vision_costmap/layers/spatio_temporal_voxel
  /zedx_vision_costmap/layers/prediction
  /zedx_vision_costmap/layers/inflation
  /zedx_vision_costmap/costmap
  /zedx_vision_costmap/diagnostics
)

available="$(ros2 topic list)"
missing=0
for topic in "${topics[@]}"; do
  if grep -Fxq "$topic" <<<"$available"; then
    type="$(ros2 topic type "$topic")"
    echo "OK $topic [$type]"
  else
    echo "MISSING $topic"
    missing=$((missing + 1))
  fi
done

if (( missing > 0 )); then
  echo "Runtime verification failed: $missing missing topic(s)."
  exit 1
fi

echo "Sampling fused costmap rate for 8 seconds..."
timeout 8 ros2 topic hz /zedx_vision_costmap/costmap || true
echo "Latest perception status:"
timeout 5 ros2 topic echo --once /zedx_vision_costmap/perception_status || true
