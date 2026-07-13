#!/usr/bin/env zsh
# Start the complete recorded-video test once and stop every child on Ctrl-C.

set -e

ROOT="${0:A:h:h}"
source /opt/ros/jazzy/setup.zsh
source "$ROOT/ros2_ws/install/setup.zsh"
cd "$ROOT"

existing=$(ros2 node list 2>/dev/null | grep -Ec '^/(perception_costmap|recorded_camera_publisher|costmap_marker_viz|three_camera_display|rviz)$' || true)
if (( existing > 0 )); then
  print -u2 "Refusing to start: $existing video-test node(s) already exist."
  print -u2 "Stop the older video-test terminal, then run this command again."
  exit 1
fi

typeset -a children

cleanup() {
  trap - EXIT
  for pid in $children; do
    kill "$pid" 2>/dev/null || true
  done
  wait $children 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' INT TERM HUP

python3 tmp/video_camera_publisher.py --scale 0.5 &
children+=($!)

costmap_executable="$(ros2 pkg prefix perception_costmap)/lib/perception_costmap/costmap_node"
"$costmap_executable" \
  --ros-args --params-file tmp/video_costmap.yaml &
children+=($!)

python3 tmp/costmap_marker_viz.py &
children+=($!)

if [[ "${VIDEO_TEST_HEADLESS:-0}" != "1" ]]; then
  python3 tmp/camera_display.py &
  children+=($!)

  rviz2 -d tmp/video_costmap_markers.rviz &
  children+=($!)
fi

print "Started one synchronized video publisher, one costmap, and marker visualization."
if [[ "${VIDEO_TEST_HEADLESS:-0}" != "1" ]]; then
  print "Started the camera display and RViz windows."
fi
print "Press Ctrl-C here to stop the complete test."
wait $children
