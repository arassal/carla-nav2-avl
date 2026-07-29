# Deploying to the car computer (Jetson AGX Orin 64 GB, "dinosaur")

> §§0–5 were written pre-deploy assuming an Orin Nano 8 GB; the real unit
> is an AGX Orin 64 GB (no swap needed, MAXN exists but needs a reboot).
> §6 records what actually happened on the hardware.

The Jetson runs the identical perception stack against real sensors. CARLA
never runs here (x86 only) — `tools/carla_feed.py` is replaced by real camera
and lidar drivers publishing the same topics.

## 0. Facts that decide everything
- Kernel 5.15.148-tegra => L4T R36.4 => JetPack 6.1 => Ubuntu 22.04 => native
  ROS2 is **Humble** — exactly our target. Install ROS2 Humble + Nav2 natively;
  no container needed. Confirm on the unit before proceeding:
  `uname -r && cat /etc/nv_tegra_release` (expect R36.x). If it reports R35.x
  (JetPack 5 / Ubuntu 20.04) instead, use a Humble container matching the
  host L4T version (e.g. dustynv/ros:humble-* for the same r35.x tag) —
  container userspace must match the host L4T major version.
- 8 GB RAM shared CPU/GPU. Add swap before building:
  `sudo fallocate -l 8G /swap && sudo mkswap /swap && sudo swapon /swap`
- Power: check modes with `sudo nvpmodel -q --verbose`, then select MAXN
  (index varies by board/JetPack — commonly `sudo nvpmodel -m 0`) and
  `sudo jetson_clocks` before benchmarks.

## 1. Torch/ultralytics (inside the container or JetPack env)
- NEVER `pip install torch` — that pulls a CPU wheel. Use NVIDIA's Jetson
  wheel matching the JetPack version (6.1 here) (developer.nvidia.com/embedded
  → PyTorch for Jetson), then `pip install ultralytics --no-deps` + its light
  deps.

## 2. Build + verify (10 min)
    cd ros2_ws && colcon build --packages-select perception_costmap
    source install/setup.bash
    cd src/perception_costmap && PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q
    python3 tools/bench_perception.py --frames 50          # hsv baseline

## 3. Models
    python3 tools/export_trt.py --weights yolov8n.pt       # ON the Jetson
    python3 tools/bench_perception.py --frames 50 --yolo-weights yolov8n.engine
    # target: yolo stage <= 25 ms (≈2x realtime headroom at 10 Hz with seg)
    # TwinLiteNet nano: benchmark with --twinlite-*; if too slow on CPU fall
    # back to hsv until a TensorRT export of it is done.

## 4. Sensors
- Cameras: v4l2_camera / the vendor driver, publishing
  /camera/front/image + camera_info (BEST_EFFORT — matches our QoS).
- Lidar: vendor ROS2 driver -> /lidar/points in base_link (or set a static TF
  and adjust lidar z band in the YAML).
- Calibrate each camera with tools/ipm_overlay.py against a tape measure on
  the ground. Do not skip this; it is the whole geometry.

## 5. Acceptance
- `ros2 topic hz /perception/costmap` >= 8 Hz with the chosen models
- RViz: road free, person standing in front = lethal within 300 ms, clears
  within 500 ms after they step away (temporal filter working)
- Nav2 local costmap (config/nav2_costmap_params.yaml) mirrors it.

## 6. As-deployed on dinosaur (2026-07-02) — what actually happened

The unit is an AGX Orin 64 GB (not a Nano), JetPack 6.1 / L4T R36.4.7,
CUDA 12.6, TensorRT 10.3. Real config: `config/perception_dinosaur.yaml`
(3x ZED X + velodyne). Ops scripts + live dashboard: `deploy/`.

**Torch (the §1 warning is real, with extra teeth):**
- Index is `https://pypi.jetson-ai-lab.io/jp6/cu126` — the old `.dev` domain
  is dead, and when it silently fails pip falls back to PyPI and installs a
  broken cu13 aarch64 wheel (`torch.cuda.is_available() == False`).
- `torch==2.8.0` + `torchvision==0.23.0`. Newer (2.11) needs `libcudss`,
  which JetPack 6.1 doesn't ship.
- Pin `numpy<2` afterwards — the wheel drags in numpy 2.x, which breaks the
  Humble cv_bridge / ultralytics import chain.

**Models, measured (30-frame bench, 30 W mode + jetson_clocks):**

    yolov8n TensorRT FP16     25.7 ms   (meets the <=25 ms target)
    TwinLiteNet+ nano, CUDA   73.7 ms   <- dominates
    3-camera node             ~5 Hz     (below the 8 Hz acceptance)

To close the gap: TensorRT-export TwinLiteNet (same treatment as YOLO), and
MAXN power mode (`nvpmodel -m 0` — requires a reboot on this board).
TwinLiteNet weights: gdown the Drive folder in the TwinLiteNetPlus README
(nano.pth = 217 KB).

**ZED X reality (§4 "vendor driver" hides all of this):**
- Wrappers block on TF ("Waiting for valid static transformations...") until
  robot_state_publisher is up — launch `avros_bringup sensors.launch.py`
  first. That also provides `/velodyne_points`: points arrive in the SENSOR
  frame, so the z band in the YAML is offset by the 0.715 m mount height
  (-0.5..1.8, not 0.2..2.5).
- Start the three cameras sequentially, ~40 s apart. Parallel starts and
  daemon restarts under live wrappers wedge the GMSL streams
  ("CAMERA STREAM FAILED TO START", Argus timeouts). Recovery that works:
  `deploy/clean_camera_restart.sh` (daemon restart + sequential bring-up).
- Topic names are `/zed_<name>/zed_node/rgb/color/rect/image` (+
  `.../camera_info`) on the current wrapper — not the older
  `rgb/image_rect_color` the docs float around.

**Still open:** per-camera IPM calibration (§4 tape-measure procedure — the
current homographies are URDF-derived) and TwinLiteNet TRT export.
Boot-time autostart shipped 2026-07-07: `deploy/percept-stack.service`
(installed + enabled on the car).
