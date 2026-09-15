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
- RAM: the real unit is an **AGX Orin 64 GB** — no swap needed. (These §0–5
  steps were first written assuming an 8 GB Orin Nano; if you really are on an
  8 GB board, add swap: `sudo fallocate -l 8G /swap && sudo mkswap /swap &&
  sudo swapon /swap`. Otherwise skip it. §6 records the real as-deployed unit.)
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
    cd src/perception_costmap && PYTHONPATH=. python3 -m pytest test -q
    python3 tools/bench_perception.py --frames 50          # hsv baseline

## 3. Models
    export AVL_MODELS_DIR="$(git rev-parse --show-toplevel)/models"   # shipped .pt weights
    python3 tools/export_trt.py --weights "$AVL_MODELS_DIR/yolov8n.pt"    # ON the Jetson
    python3 tools/export_trt.py --weights "$AVL_MODELS_DIR/cone_det.pt"
    python3 tools/bench_perception.py --frames 50 --yolo-weights "$AVL_MODELS_DIR/yolov8n.engine"
    # target: yolo stage <= 25 ms (≈2x realtime headroom at 10 Hz with seg)

### Enabling TwinLiteNet (optional — the default is hsv)
`export_trt.py` only builds YOLO engines. TwinLiteNet road segmentation is
NOT turn-key and the shipped default is `segmentation_method: hsv`. To enable
the better segmenter:
    git clone https://github.com/chequanghuy/TwinLiteNetPlus \
        "$AVL_MODELS_DIR/TwinLiteNetPlus"          # provides the model class
    # nano.pth is shipped at $AVL_MODELS_DIR/twinlite_nano.pth; build a
    # TensorRT engine from it (ONNX export -> trtexec, FP16, input 384x640,
    # output order img -> (drivable_area, lane_line)) and save it as
    # $AVL_MODELS_DIR/twinlite_nano.engine
    # then set segmentation_method: twinlitenet in config/perception_dinosaur.yaml
Until a real exporter is committed, this remains a manual step; without a valid
`.engine` the node logs a warning and runs hsv.

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
- Nav2: bring it up with `deploy/real_nav2_launch.py` (uses
  `config/real_nav2_params.yaml` and auto-starts the obstacle-cloud bridge).
  Confirm the local costmap mirrors the perception costmap.
  NOTE: `config/nav2_costmap_params.yaml` is deprecated/reference-only — do not
  use it (it feeds a base_link grid to StaticLayer and freezes; see its header).

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

> **Historical (2026-07-02).** The car has since switched to
> `segmentation_method: hsv` (`config/perception_dinosaur.yaml`), so
> TwinLiteNet is no longer in the loop and no longer the bottleneck. A laptop
> profile (ISSUES.md P2) now points at the node's own main thread. Re-measure
> on the car with §7 before acting on either number.

To close the gap (as planned then): TensorRT-export TwinLiteNet (same treatment as YOLO), and
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
current homographies are URDF-derived). TwinLiteNet TRT export only matters if
TwinLiteNet comes back into the car config.
Boot-time autostart shipped 2026-07-07: `deploy/percept-stack.service`
(installed + enabled on the car).

## 7. Car test checklist (PRs #1-#4, 2026-09-14)

Everything below has been tested on a laptop only. Run it on dinosaur before
merging the lean launch into the boot service, and **before any costmap
performance work** (ISSUES.md P2 is on hold until this is done). Record
results in `logs/results/<date>_car-test.md`.

In every terminal, match the boot stack first (login shells default to domain 42):
`export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` plus `CYCLONEDDS_URI`
as in `deploy/full_stack_restart.sh`.

**0. Baseline, on the current boot stack (before changing anything)**

```bash
ros2 topic hz /perception/costmap            # costmap rate
tegrastats --interval 1000                   # CPU/GPU load, ~30 s
ros2 topic list | grep confidence            # expect 3 confidence_map topics (ISSUES.md C9)
```

**1. Map click (PR #1).** Start autodrive the usual way (`auto_drive.launch.py`
from avros_bringup), click a destination in RViz. Expected: no "computer-vision Nav2 bridge is stale";
`ros2 topic hz /perception/costmap_cloud` about 10 Hz.

**2. Costmap node (PR #3).** After `colcon build` with PR #3, stop the boot
stack's own costmap first (Ctrl-C in its window: `tmux -L percept attach -t percept`,
window `costmap`) so two nodes don't publish the same topic. From the package directory:
```bash
ros2 launch perception_costmap perception.launch.py \
    config:=$(ros2 pkg prefix perception_costmap)/share/perception_costmap/config/perception_dinosaur.yaml
ros2 topic echo /diagnostics --once          # camera / detectors / output status
deploy/fresh_run.sh --perception             # reset (Nav2 not running): expect exit 0
```

**3. Lean launch (PR #4).** Stop the boot stack first
(`sudo systemctl stop percept-stack`), then:
```bash
ros2 launch perception_costmap perception_stack.launch.py
ros2 topic list | grep zed_                  # expect only rgb, depth, confidence per camera
ros2 topic hz /perception/costmap
tegrastats --interval 1000                   # compare with step 0
```
In a second terminal, run the same launch again: it should log "already
running -- reusing" and start nothing.

**4. Camera sides (ISSUES.md C8).** Cover the LEFT camera by hand and watch
`ros2 topic hz /zed_left/zed_node/rgb/color/rect/image` vs `/zed_right/...`:
the image that goes dark tells you which serial is really on the left.

**5. Depth quality.** The lean profiles set `depth_stabilization: 0`. In RViz,
view `/zed_front/zed_node/depth/depth_registered` with the robot parked facing
a wall or cone. If it flickers badly, set `depth_stabilization: 1` and
`pos_tracking_enabled: true` in `config/zed_perception_*.yaml` and repeat step 3.

**What to bring back:** costmap Hz and tegrastats for steps 0 and 3, the
confidence topic list, which camera went dark in step 4, and any launch errors.

| check | result | notes |
|---|---|---|
| 0. baseline costmap Hz / CPU / GPU | | |
| 0. confidence topics present? | | |
| 1. click accepted, cloud Hz | | |
| 2. costmap publishes, reset clean | | |
| 3. lean launch Hz / CPU / GPU | | |
| 3. second launch reuses | | |
| 4. real left serial | | |
| 5. depth OK with stabilization 0? | | |
