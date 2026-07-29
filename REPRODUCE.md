# Reproducing the perception + obstacle-avoidance stack

Everything needed to run this on the same vehicle is in this repo: source,
model weights (`models/`, `.pt`/`.pth`), the TensorRT export script, configs,
and launch files. The one thing that cannot be shipped as a file is the
TensorRT `.engine` — it is compiled per GPU/CUDA/TensorRT version and must be
built on the target machine (step 4).

## Hardware assumed (the "same car")
- NVIDIA Jetson AGX Orin (JetPack 6.1 / L4T R36.4, Ubuntu 22.04, ROS 2 Humble)
- 3x Stereolabs ZED X cameras (front/left/right, GMSL)
- Velodyne VLP-16 lidar
- Teensy 4.1 -> 2x REV SparkMAX over CAN (differential/skid-steer base)
- Xsens MTi-680G IMU/GNSS

## Steps
1. **Clone** this repo.

2. **Install dependencies** — follow `ros2_ws/src/perception_costmap/DEPLOY.md`
   (torch/ultralytics Jetson wheels, TensorRT, ZED SDK, ROS 2 Humble + Nav2).
   The §1 torch warning is real; read it.

3. **Point at the models:**
   ```bash
   export AVL_MODELS_DIR="$(pwd)/models"
   ```
   `config/perception_dinosaur.yaml` references weights as `${AVL_MODELS_DIR}/...`
   and `costmap_node.py` expands that at load time.

4. **Build the TensorRT engines** from the committed `.pt` weights (on target):
   ```bash
   cd ros2_ws/src/perception_costmap
   python3 tools/export_trt.py --weights "$AVL_MODELS_DIR/yolov8n.pt"
   python3 tools/export_trt.py --weights "$AVL_MODELS_DIR/cone_det.pt"
   # -> yolov8n.engine / cone_det.engine next to the .pt files
   ```
   (Until an engine exists, the node falls back to the `.pt` / hsv per DEPLOY.md.)

5. **Build the workspace:**
   ```bash
   cd ros2_ws && colcon build && source install/setup.bash
   ```

6. **Sensor bring-up (PREREQUISITE, separate repo).** Cameras, lidar, IMU,
   TF, and EKF come from our avros bringup, which is **not** in this repo:
   https://github.com/Paarseus/IGVC_ROS2 — `ros2 launch avros_bringup
   sensors.launch.py` plus the ZED wrappers. This publishes the topics the
   costmap subscribes to (`/zed_*/zed_node/rgb/color/rect/image`,
   `/velodyne_points`, `odom`→`base_link` TF). Without it there is no sensor
   data and nothing downstream runs.

7. **Run the costmap + Nav2:**
   ```bash
   # perception costmap:
   ros2 launch perception_costmap perception.launch.py \
        config:=src/perception_costmap/config/perception_dinosaur.yaml
   # Nav2 (this launch ALSO auto-starts the obstacle-cloud bridge that feeds
   # Nav2's ObstacleLayer -- you do not start it separately):
   ros2 launch src/perception_costmap/deploy/real_nav2_launch.py
   ```

## Notes
- `deploy/*.sh` are the **as-run operational scripts from our car** and contain
  host-specific absolute paths (`/home/dinosaur/...`). Treat them as reference,
  not portable entry points — the launch files + config above are the portable
  path.
- Nav2 is **disarmed by default**: its `/cmd_vel` is remapped to `/nav2/cmd_vel`
  so it cannot drive the motors. Set `NAV2_ARMED=1` to connect it to the real
  `/cmd_vel` (see `deploy/real_nav2_launch.py`).
- `deploy/plan_then_go.py` verifies a planned path clears detected obstacles
  before executing — use it rather than sending raw goals during bring-up.
