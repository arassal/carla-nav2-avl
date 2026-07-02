# Deploying to the car computer (Jetson Orin Nano, "dinosaur")

The Jetson runs the identical perception stack against real sensors. CARLA
never runs here (x86 only) — `tools/carla_feed.py` is replaced by real camera
and lidar drivers publishing the same topics.

## 0. Facts that decide everything
- Kernel 5.15.148-tegra => JetPack 5.x => Ubuntu 20.04 => native ROS2 is
  **Foxy**. Our target is Humble. Two options:
  a) (recommended) run the stack in a Humble container:
     `dustynv/ros:humble-desktop-l4t-r35.4.1` with `--runtime nvidia`, or
  b) build on Foxy natively — this package avoids Humble-only APIs, but Nav2
     Foxy is EOL; prefer (a).
- 8 GB RAM shared CPU/GPU. Add swap before building: 
  `sudo fallocate -l 8G /swap && sudo mkswap /swap && sudo swapon /swap`
- Power: `sudo nvpmodel -m 0 && sudo jetson_clocks` (MAXN) before benchmarks.

## 1. Torch/ultralytics (inside the container or JetPack env)
- NEVER `pip install torch` — that pulls a CPU wheel. Use NVIDIA's Jetson
  wheel matching the JetPack version (developer.nvidia.com/embedded → PyTorch
  for Jetson), then `pip install ultralytics --no-deps` + its light deps.

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
