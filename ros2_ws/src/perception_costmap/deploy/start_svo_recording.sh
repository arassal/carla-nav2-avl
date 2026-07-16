#!/bin/bash
# Start SVO recording on all 3 live ZED X cameras simultaneously.
#
# SVO captures the raw stereo pair (pre-rectification) + depth/IMU/odom in one
# file per camera, H.265 compressed. Unlike the old rosbag-of-rect_image
# approach, depth/pose/etc are reprocessable offline at different SDK
# settings later -- nothing is baked in at record time. This is a passive tap
# on each zed_node's existing capture loop (start_svo_rec / stop_svo_rec
# services, already exposed by zed_wrapper) -- it does NOT restart, pause, or
# otherwise disturb the live camera stream or downstream perception/costmap
# pipeline.
#
# Usage: start_svo_recording.sh [output_dir] [bitrate_kbps]
set -e
OUTDIR="${1:-/home/dinosaur/svo_recordings/svo_$(date +%Y%m%d_%H%M%S)}"
BITRATE="${2:-12000}"
mkdir -p "$OUTDIR"

source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

for cam in front left right; do
  echo "Starting SVO recording: $cam -> $OUTDIR/$cam.svo2"
  ros2 service call /zed_${cam}/zed_node/start_svo_rec zed_msgs/srv/StartSvoRec \
    "{bitrate: $BITRATE, compression_mode: 2, target_framerate: 0, input_transcode: false, svo_filename: $OUTDIR/${cam}.svo2}" \
    | grep -E 'success|message'
done

echo "$OUTDIR" > /tmp/svo_recording_current_dir
echo
echo "Recording to: $OUTDIR"
echo "Stop with: bash stop_svo_recording.sh"
