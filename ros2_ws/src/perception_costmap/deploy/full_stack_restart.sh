#!/bin/bash
# Rebuild the whole percept tmux -L percept session from scratch: sensors (TF+lidar),
# 3 ZED cameras (sequential, watchdog loops), fused costmap (TRT engine +
# TwinLiteNet), live viz node, web_video_server, dashboard http server.
#
# Usage:
#   full_stack_restart.sh          # manual restart: wrap tmux in a systemd
#                                   # --user --scope so it survives SSH teardown
#   full_stack_restart.sh --boot    # boot-time restart (invoked by
#                                   # percept-stack.service): skip the scope
#                                   # wrapper -- the service's own cgroup is
#                                   # the persistence
set -e

BOOT=false
if [ "$1" = "--boot" ]; then
  BOOT=true
fi

CFG=/home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config
E="export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI=file://$CFG/cyclonedds.xml"

tmux -L percept kill-session -t percept 2>/dev/null || true
pkill -f "zed_camera.launch.py" 2>/dev/null || true
pkill -f costmap_node 2>/dev/null || true
pkill -f "python3 .*viz_node\.py" 2>/dev/null || true
pkill -f web_video_server 2>/dev/null || true
pkill -f "python3 -m http\.server 8090" 2>/dev/null || true
sleep 3

echo "[1/6] sensors (TF + velodyne + xsens)..."
# Start the tmux server in a systemd user scope (linger is enabled): a scope
# lives until every process in it exits, so the daemonized tmux server
# survives SSH teardown. (A service unit reaps it the moment the client
# detaches — learned the hard way.) Dedicated socket (-L percept): on the
# default socket an operator's pre-existing tmux server would own the
# session in THEIR login cgroup, and the scope would protect nothing.
# --boot (percept-stack.service): the service cgroup is the persistence --
# plain tmux, no scope wrapper.
if [ "$BOOT" = false ]; then
  systemctl --user stop percept-tmux.scope 2>/dev/null || true
  systemctl --user reset-failed percept-tmux.scope 2>/dev/null || true
  systemd-run --user --scope --collect --unit percept-tmux \
    tmux -L percept new-session -d -s percept -n sensors \
    "bash -c \"source /home/dinosaur/IGVC/install/setup.bash && $E && ros2 launch avros_bringup sensors.launch.py 2>&1 | tee /tmp/sensors.log; exec bash\""
else
  tmux -L percept new-session -d -s percept -n sensors \
    "bash -c \"source /home/dinosaur/IGVC/install/setup.bash && $E && ros2 launch avros_bringup sensors.launch.py 2>&1 | tee /tmp/sensors.log; exec bash\""
fi
sleep 3
tmux -L percept has-session -t percept
sleep 15

echo "[1b/6] EKF (odom->base_link; velodyne + Nav2 need the odom frame)..."
tmux -L percept new-window -t percept -n ekf \
  "bash -c \"source /home/dinosaur/IGVC/install/setup.bash && $E && ros2 run robot_localization ekf_node --ros-args -r __node:=ekf_filter_node_odom --params-file $CFG/ekf.yaml 2>&1 | tee /tmp/ekf.log; exec bash\""
sleep 5

zed_cmd() {
  local name=$1 serial=$2
  echo "source /home/dinosaur/IGVC/install/setup.bash && $E && \
while true; do \
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx \
camera_name:=$name serial_number:=$serial publish_tf:=false \
publish_urdf:=false ros_params_override_path:=$CFG/${name}.yaml \
2>&1 | tee -a /tmp/${name}.log; \
echo \"[watchdog] $name exited, relaunching in 5s\" | tee -a /tmp/${name}.log; \
sleep 5; done"
}

echo "[2/6] zed_front..."
tmux -L percept new-window -t percept -n zed_front "bash -c '$(zed_cmd zed_front 42569280)'"
sleep 40

echo "[3/6] zed_left..."
tmux -L percept new-window -t percept -n zed_left "bash -c '$(zed_cmd zed_left 49910017)'"
sleep 40

echo "[4/6] zed_right..."
tmux -L percept new-window -t percept -n zed_right "bash -c '$(zed_cmd zed_right 43779087)'"
sleep 40

echo "[5/6] costmap (3-cam fused, TRT engine + TwinLiteNet)..."
tmux -L percept new-window -t percept -n costmap \
  "bash -c \"source /opt/ros/humble/setup.bash && source /home/dinosaur/carla-nav2-avl/ros2_ws/install/setup.bash && $E && ros2 launch perception_costmap perception.launch.py config:=/home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/config/perception_dinosaur.yaml 2>&1 | tee /tmp/costmap.log; exec bash\""
sleep 15

echo "[6/6] viz node + streaming servers..."
tmux -L percept new-window -t percept -n viz \
  "bash /home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/deploy/run_viz.sh 2>&1 | tee /tmp/viz_node.log"
tmux -L percept new-window -t percept -n wvs \
  "bash -c \"source /opt/ros/humble/setup.bash && $E && ros2 run web_video_server web_video_server --ros-args -p port:=8080 -p address:=0.0.0.0 2>&1 | tee /tmp/wvs.log; exec bash\""
tmux -L percept new-window -t percept -n www \
  "bash -c \"cd /home/dinosaur/live_dashboard && python3 -m http.server 8090 --bind 0.0.0.0; exec bash\""

echo "done: $(tmux -L percept list-windows -t percept -F '#W' | tr '\n' ' ')"
