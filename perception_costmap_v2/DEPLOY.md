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

`carla_feed.py` puts the server in **synchronous mode at 20 Hz** (matching
the lidar's `rotation_frequency`) so that one tick equals one full lidar
sweep. Do not remove this without reading the next section.

### Why the feed forces synchronous mode

A free-running CARLA server ticks at whatever framerate it manages (~100 Hz
here). CARLA slices a lidar sweep across the ticks it spans, so a 20 Hz lidar
on a 100 Hz server delivers a *fraction of a rotation* per callback --
and `costmap_node.on_lidar` replaces its point set rather than accumulating,
so the costmap would only ever hold that wedge. Obstacles would appear and
then be cleared a few tens of milliseconds later as the wedge swept past.
Measured on Town10HD_Opt (2026-07-31), binning each message's points into 36
azimuth bins:

| mode | points/msg | azimuth bins covered |
|---|---|---|
| `--async` (free-running) | 289-385 | 9-11 / 36 |
| synchronous (default) | 1249-1339 | **36 / 36** |

Two consequences to know about:

- **It is global server state.** The feed saves the previous settings and
  restores them (plus the traffic manager's sync flag) on exit. If it is
  ever killed with `SIGKILL`, other CARLA clients on the box will hang
  waiting for ticks nobody sends; reconnect and set
  `synchronous_mode = False` to unstick them.
- **The feed becomes the only clock.** Nothing advances unless this process
  calls `world.tick()`, so don't run two tick-driving clients at once.

`--async` restores the old behaviour if you specifically want it (e.g. to
share the server with another tool that drives the clock).

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

The `--fx 500 --fy 500 --pitch 12` in this tool's docstring are
placeholders, NOT the values for `carla_feed.py`'s camera. Derive them:

- `carla_feed.py` uses 640x480 at FOV 90 with no Rotation, so
  `fx = fy = (W/2)/tan(FOV/2) = 320`, `cx = 320`, `cy = 240`, `pitch = 0`.
- `--cam-x 1.5` -- the camera's forward mount offset, or every distance is
  off by 1.5 m.
- `--cam-height` is the camera's height **above the road surface**, which is
  NOT the 1.6 in `CAMERA_MOUNT_Z`. See below.

```bash
PYTHONPATH=. python3 tools/ipm_overlay.py --image /tmp/carla_eval_frames/rgb_000010.png \
    --fx 320 --fy 320 --cx 320 --cy 240 --cam-height 1.939 --cam-x 1.5 --pitch 0 \
    --out /tmp/overlay.png
```

### cam-height is measured from the road, not from the vehicle origin

A CARLA vehicle's actor origin floats above the road. Sensors mount
relative to that origin, so a camera at `Location(z=1.6)` sits ~1.9 m above
the road, not 1.6 m. Feeding the mount offset instead of the true height is
a silent systematic error:

| `--cam-height` | max reprojection error vs CARLA ground truth |
|---|---|
| 1.600 (mount offset) | 43.38 px |
| 1.939 (true height, that run) | 0.00 px |

**Do not copy 1.939.** The origin offset depends on the vehicle blueprint
and on how far the suspension has settled -- across runs of the same
blueprint on Town10HD_Opt it measured 0.275, 0.307, 0.338, 0.339 m, i.e.
cam-height 1.875 to 1.939. Measure it in the same session you calibrate in:

```python
tf = ego.get_transform(); ct = cam.get_transform()
road_z = world.get_map().get_waypoint(tf.location, project_to_road=True).transform.location.z
print(ct.location.z - road_z)      # <-- this is --cam-height
```

The same offset applies to `LIDAR_MOUNT_Z = 1.8`, which feeds
`carla_lidar_to_rep103(sensor_z=...)`.

Verified 2026-07-30 on Town10HD_Opt: with the true height, the homography
reproduces CARLA's own projection of 9 ground probes (5-16 m, +/-3 m
lateral) to 0.00 px on both axes, including the left-handed-to-REP-103
lateral sign. The homography implementation itself is correct; only the
height parameter was ever in question.

Open `/tmp/overlay.png` and confirm the drawn 1m grid lines land where a
tape measure would put them. Do this for every camera before trusting
anything downstream -- a bad homography poisons the whole costmap and is
invisible in RViz (the road looks fine, obstacles look fine, they're just
all in the wrong place). The overlay is a human sanity check; the probe
comparison above is the objective one, and is worth scripting for a real
camera whenever ground-truth geometry is available.

## 4. Measure segmenter accuracy

```bash
PYTHONPATH=. python3 tools/eval_road_iou.py --dump-dir /tmp/carla_eval_frames \
    --methods hsv --road-tags 1 24
```

The geometry flags (`--fx --cam-height --cam-x --pitch`) default to
`carla_feed.py`'s camera and feed the BEV column's homography. They used to
default to the same `fx 500 / pitch 12` placeholders section 3 warns about,
which scored BEV against a homography no frame was ever taken with; on the
frames below that alone read 0.2653 instead of 0.7511. **If your frames came
from a different camera, pass its real values** -- and re-read section 3 on
`--cam-height` being measured from the road.

Measured on 40 Town10HD_Opt frame pairs (2026-07-31):

| method | IoU (image) | IoU (BEV) |
|---|---|---|
| hsv | 0.3430 | 0.7511 |

The image-space number is low for a reason worth knowing before you trust
`hsv` on a real camera: precision 0.567 / recall 0.430, and the false
positives are almost entirely **above the horizon**. The grey sky and
buildings of Town10HD fall inside the `(0,0,60)-(180,60,200)` HSV band and
connect to the road through the horizon line, so `_largest_blob` returns one
blob spanning both -- 41% of the top half is predicted road where ground
truth is 0%. Restricting to below the horizon before blob selection lifts
image IoU 0.343 -> 0.451.

This does not corrupt the costmap today, which is why BEV scores so much
higher: `bev_known_mask` only samples the ground footprint, so the sky
detections are discarded before they reach a cell. It does mean the image
IoU understates `hsv` and that anything which bypasses that mask (or any
camera whose horizon isn't at `cy`) inherits the bug.

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
