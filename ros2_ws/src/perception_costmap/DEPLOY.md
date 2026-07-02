# Deploying to the car computer (Jetson Orin Nano, "dinosaur")

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
