#!/bin/bash
# Full clean camera restart, mimicking the team's orin_launch_all.sh:
# stop everything camera-related, restart both camera daemons, then bring
# up the three wrappers SEQUENTIALLY (watchdog loops included).
CFG=/home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config

# no percept session (fresh boot / full crash)? -> full restart instead
if ! tmux -L percept has-session -t percept 2>/dev/null; then
  echo "no percept session; running full_stack_restart.sh instead"
  exec bash "$(dirname "$0")/full_stack_restart.sh"
fi

echo "[1/4] stopping camera windows..."
for w in zed_front zed_left zed_right; do
  tmux -L percept kill-window -t percept:$w 2>/dev/null
done
pkill -f "zed_camera.launch.py" 2>/dev/null
sleep 5

echo "[2/4] restarting camera daemons..."
sudo systemctl restart nvargus-daemon 2>/dev/null
sudo systemctl restart zed_x_daemon
sleep 10

zed_cmd() {
  local name=$1 serial=$2
  echo "source /home/dinosaur/IGVC/install/setup.bash && \
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
export CYCLONEDDS_URI=file://$CFG/cyclonedds.xml && \
while true; do \
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx \
camera_name:=$name serial_number:=$serial publish_tf:=false \
publish_urdf:=false ros_params_override_path:=$CFG/${name}.yaml \
2>&1 | tee -a /tmp/${name}.log; \
echo \"[watchdog] $name exited, relaunching in 5s\" | tee -a /tmp/${name}.log; \
sleep 5; done"
}

echo "[3/4] starting cameras sequentially (40s apart)..."
tmux -L percept new-window -t percept -n zed_front "bash -c '$(zed_cmd zed_front 42569280)'"
sleep 40
tmux -L percept new-window -t percept -n zed_left "bash -c '$(zed_cmd zed_left 49910017)'"
sleep 40
tmux -L percept new-window -t percept -n zed_right "bash -c '$(zed_cmd zed_right 43779087)'"
sleep 40

echo "[4/4] done"
