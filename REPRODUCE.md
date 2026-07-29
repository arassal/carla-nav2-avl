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

6. **Run.** Sensors + cameras come from the IGVC/avros bringup on the car
   (`sensors.launch.py`, ZED wrappers); this package adds the costmap and the
   Nav2 obstacle-avoidance layer:
   ```bash
   # perception costmap:
   ros2 launch perception_costmap perception.launch.py \
        config:=src/perception_costmap/config/perception_dinosaur.yaml
   # obstacle cloud + Nav2 (see deploy/):
   python3 src/perception_costmap/deploy/costmap_to_cloud.py &
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
