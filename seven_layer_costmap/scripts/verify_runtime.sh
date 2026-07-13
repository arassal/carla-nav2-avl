#!/usr/bin/env bash
set -euo pipefail

topics=(
  /seven_layer_costmap/layers/lanelet
  /seven_layer_costmap/layers/static_obstacle
  /seven_layer_costmap/layers/spatio_temporal_voxel
  /seven_layer_costmap/layers/prediction
  /seven_layer_costmap/layers/inflation
  /seven_layer_costmap/layers/traffic_regulation
  /seven_layer_costmap/layers/road_condition
  /seven_layer_costmap/costmap
  /seven_layer_costmap/diagnostics
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
timeout 8 ros2 topic hz /seven_layer_costmap/costmap || true
echo "Latest perception status:"
timeout 5 ros2 topic echo --once /seven_layer_costmap/perception_status || true
