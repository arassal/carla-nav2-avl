#!/bin/bash
# Stop SVO recording started by start_svo_recording.sh on all 3 cameras.
set -e
source /opt/ros/humble/setup.bash
source /home/dinosaur/IGVC/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

for cam in front left right; do
  echo "Stopping SVO recording: $cam"
  ros2 service call /zed_${cam}/zed_node/stop_svo_rec std_srvs/srv/Trigger "{}" | grep -E 'success|message'
done

if [ -f /tmp/svo_recording_current_dir ]; then
  DIR=$(cat /tmp/svo_recording_current_dir)
  echo
  echo '--- recorded files ---'
  ls -la "$DIR" 2>/dev/null
fi
