# Deploying to a real CARLA box / real car computer

This package was built and tested in a sandbox with **no GPU** (2 CPU
cores, no CARLA). Every algorithm module (`occupancy.py`, `bev.py`,
`temporal.py`, `obstacles.py`, `segmentation.py`, `carla_convert.py`) is
verified there via the offline pytest suite and `tools/synthetic_demo.py`
(synthetic camera+lidar data standing in for CARLA). This document is what
to do next on a machine that actually has CARLA and/or a GPU.

## 0. What runs where

| Component | Needs | Runs on |
|---|---|---|
| Core algorithm modules + pytest suite | numpy, opencv only | anywhere (already verified here) |
| `tools/synthetic_demo.py` | + matplotlib | anywhere (already verified here) |
| `tools/carla_feed.py` | carla==0.9.16, rclpy | the CARLA/GPU box only |
| `tools/eval_road_iou.py` | opencv (reads dumped PNGs) | anywhere, but frames come from the CARLA box |
| `tools/ipm_overlay.py` | opencv only | anywhere (works on any single frame) |
| `costmap_node.py` (`attach_ros`/`main`) | rclpy, sensor_msgs, nav_msgs | wherever ROS2 + the sensors are (CARLA box or Jetson) |
| `tools/export_trt.py` | ultralytics, TensorRT | ON the target device only (engines are hardware-specific) |

## 1. First run on your CARLA/5090 machine

```bash
# unzip this project next to (or inside) your existing carla-nav2-avl
# checkout, e.g. as a new top-level directory: perception_costmap_v2/
cd perception_costmap_v2
pip install -r requirements.txt --break-system-packages   # or a venv

# confirm the ported algorithm still passes on real hardware (should match
# the 68 passed / 0 failed we got in the no-GPU sandbox)
PYTHONPATH=. python3 -m pytest test -q

# if a ROS2 environment is sourced, pytest auto-loads ROS's launch_testing
# plugins and dies before collection ("No module named 'lark'", or
# "unknown hook 'pytest_launch_collect_makemodule'"). That is the ambient
# ROS install, not this suite -- disable the entrypoint:
PYTHONPATH=. python3 -m pytest test -q -p no:launch_testing_ros_pytest_entrypoint

# confirm the synthetic demo still renders (sanity check before touching
# real CARLA -- if this fails, something about YOUR environment differs
# from the sandbox, not CARLA)
PYTHONPATH=. python3 tools/synthetic_demo.py
```

## 2. Connect real CARLA

```bash
# terminal 1: start the simulator
DISPLAY=:0 ~/CARLA_0.9.16/CarlaUE4.sh -windowed -ResX=960 -ResY=540

# terminal 2: source ROS2, then run the feed
source /opt/ros/jazzy/setup.bash
PYTHONNOUSERSITE=1 PYTHONPATH=.:$PYTHONPATH python3 tools/carla_feed.py \
    --host 127.0.0.1 --port 2000 \
    --dump-dir /tmp/carla_eval_frames --dump-every 10
```

### Environment gotchas on the AVL sim box (verified 2026-07-30)

These cost an hour the first time; none of them are bugs in this package.

- **Do not pass `-quality-level=Low` or `-opengl`.** UE4 crashes at startup
  with `FUnixPlatformMisc::RequestExitWithStatus` (shader compile failure).
  For low GPU load, set `settings.no_rendering_mode = True` through the
  Python API after connecting instead.
- **ROS2 here is Jazzy, not Humble** -- `/opt/ros/jazzy/setup.bash`.
- **Use `/usr/bin/python3` (3.12), not the conda python3** on `PATH`
  (3.13). rclpy's C extension and the `carla` wheel are only installed for
  the system interpreter.
- **`PYTHONPATH=.` alone breaks rclpy.** It replaces rather than prepends,
  so ROS2's site-packages drop off `sys.path` and `carla_feed.py` exits with
  "rclpy/sensor_msgs not importable". Always `PYTHONPATH=.:$PYTHONPATH`.
- **numpy conflict on the `--dump-dir` path.** `~/.local` has numpy 2.5.1,
  which shadows the system numpy 1.26.4 that the system cv2 4.6.0 was
  compiled against, so `import cv2` inside the lidar callback dies with
  "numpy.core.multiarray failed to import" -- publishing keeps working and
  only the PNG dump silently fails. `PYTHONNOUSERSITE=1` selects the
  consistent system pair; note it also hides the `carla` wheel (it lives in
  `~/.local`), so symlink `carla` and `carla.libs` into a scratch directory
  and add that to `PYTHONPATH`.

Watch the console for the first-frame line:

```
semantic tags observed this frame: [...]
```

Verify those ids actually correspond to Roads/RoadLines before trusting
`carla_convert.semantic_to_road_mask`'s default `road_tags=(1, 24)` --
CARLA's tag ids have changed across versions before. If the line scrolls
past, recover the same information from the dumped frames:

```bash
python3 -c "import cv2,numpy as np,glob; \
im=cv2.imread(sorted(glob.glob('/tmp/carla_eval_frames/sem_*.png'))[0]); \
print(np.unique(im[:,:,2]))"
```

Measured on Town10HD_Opt with CARLA 0.9.16 (2026-07-30): tag 1 covered
43.4% of the first frame and tag 24 covered 2.4%, i.e. the `(1, 24)`
default is correct for this server build.

## 3. Calibrate each camera

```bash
PYTHONPATH=. python3 tools/ipm_overlay.py --image /tmp/carla_eval_frames/rgb_000010.png \
    --fx 500 --fy 500 --cx 320 --cy 240 --cam-height 1.6 --pitch 12 \
    --out /tmp/overlay.png
```

Open `/tmp/overlay.png` and confirm the drawn 1m grid lines land where a
tape measure would put them. Do this for every camera before trusting
anything downstream -- a bad homography poisons the whole costmap and is
invisible in RViz (the road looks fine, obstacles look fine, they're just
all in the wrong place).

## 4. Measure segmenter accuracy

```bash
PYTHONPATH=. python3 tools/eval_road_iou.py --dump-dir /tmp/carla_eval_frames \
    --methods hsv --road-tags 1 24
```

Add `twinlitenet` to `--methods` once you have `nano.pth` weights
(`segmentation_method: twinlitenet` in the config); it degrades to `hsv`
automatically if the weights don't load, so it's always safe to try.

## 5. Run the real ROS2 node

```bash
cd ros2_ws   # if you've wired this into a colcon workspace
colcon build --packages-select perception_costmap
source install/setup.bash
ros2 run perception_costmap costmap_node --ros-args --params-file config/perception_costmap.yaml
```

`attach_ros()` in `costmap_node.py` currently wires only the publisher --
per-camera/lidar subscriptions (topic names + cv_bridge conversion) are
the one piece that genuinely needs a live ROS graph to build out; that's
the next task on a machine where `ros2 topic list` actually shows
something.

## 6. Nav2 acceptance checklist

- `ros2 topic hz /perception/costmap` >= 8 Hz
- RViz: road reads free, a pedestrian standing in view goes lethal within
  ~300ms and clears within ~500ms after stepping away (temporal filter
  working -- see `test/test_costmap_node.py` and the synthetic demo for
  the same behavior proven offline)
- Nav2's local costmap (`config/nav2_costmap_params.yaml`) visibly mirrors
  `/perception/costmap`

## 7. Jetson-specific notes (only if targeting real hardware)

- Never `pip install torch` on the Jetson -- it pulls a CPU wheel. Use
  NVIDIA's Jetson-matched wheel, then `pip install ultralytics --no-deps`.
- `tools/export_trt.py` must run ON the Jetson -- a `.engine` built on the
  CARLA/dev box will not load on different silicon.
- `python3 tools/bench_perception.py --frames 50 --yolo-weights
  yolov8n.engine` -- re-run this on-device; the numbers from this sandbox
  (2 CPU cores, no GPU, ~300 Hz on the classical/HSV path) do not transfer.
