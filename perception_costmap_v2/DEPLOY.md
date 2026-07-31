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
- `--cam-height` is the camera's height **above the road surface**. For a
  settled CARLA vehicle that is just `CAMERA_MOUNT_Z` -- but only once it
  has settled. See below.

```bash
PYTHONPATH=. python3 tools/ipm_overlay.py --image /tmp/carla_eval_frames/rgb_000010.png \
    --fx 320 --fy 320 --cx 320 --cy 240 --cam-height 1.6 --cam-x 1.5 --pitch 0 \
    --out /tmp/overlay.png
```

Verify it objectively rather than by eye:

```bash
PYTHONPATH=. python3 tools/carla_calib_check.py
```

Verified 2026-07-31 on Town10HD_Opt: with `--cam-height 1.6`, the homography
reproduces CARLA's own projection of 9 ground probes (5-16 m, +/-3 m
lateral) to **0.32 px**, including the left-handed-to-REP-103 lateral sign.
`homography_from_extrinsics` is correct.

### Let the vehicle settle before measuring anything

CARLA spawn points sit **0.6 m above the road** so a vehicle can never spawn
embedded in it. The car then falls and settles. Sampling the transform
mid-fall reads a transient as if it were a permanent mount offset, and the
transient is big enough to look plausible. Measured with a 0.05 s fixed
timestep:

| ticks after spawn (sim time) | actor origin above road |
|---|---|
| 0 | +0.000 m (spawn height, still 0.6 m up in world z) |
| 5 (0.25 s) | +0.282 m |
| 10 (0.50 s) | -0.017 m |
| 20+ (1.0 s+) | **-0.007 m** (steady state) |

So a settled `vehicle.*` origin is level with the road to within 7 mm, and
`Location(z=1.6)` really is 1.6 m up. An earlier revision of this document
claimed the origin floats ~0.3 m and that `--cam-height` had to be ~1.9;
that number was sampled at roughly the 0.25 s row above and is wrong.
`tools/carla_calib_check.py` now ticks synchronously for 60 steps before
measuring, which is why its residual is 0.32 px at 1.600.

If you still want to measure it yourself, do it after settling:

```python
tf = ego.get_transform(); ct = cam.get_transform()
road_z = world.get_map().get_waypoint(tf.location, project_to_road=True).transform.location.z
print(ct.location.z - road_z)      # <-- this is --cam-height
```

The same reasoning applies to `LIDAR_MOUNT_Z = 1.8` feeding
`carla_lidar_to_rep103(sensor_z=...)`: measured on a settled vehicle the
true lidar height is 1.793 m, so the shipped 1.8 is right and
`remove_ground_plane`'s `z_min = -0.3` band keeps its full margin. (Had the
0.3 m offset been real it would have been serious -- ground returns would
land at exactly z = -0.3, right on the rejection boundary, and half the road
surface would have entered the costmap as lethal obstacles.)

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
frames below that alone read 0.2653 instead of 0.7923. **If your frames came
from a different camera, pass its real values** -- and re-read section 3 on
letting the vehicle settle before trusting a measured `--cam-height`.

Measured on 40 Town10HD_Opt frame pairs (2026-07-31):

| method | IoU (image) | IoU (BEV) |
|---|---|---|
| hsv | 0.3430 | 0.7923 |

(The BEV column is itself a calibration check: the same frames score 0.7511
at `--cam-height 1.9` and 0.2653 at the old placeholders, so a wrong
homography shows up here as a worse segmenter score.)

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

Or, without a colcon workspace (which is how it was brought up here):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=.:$PYTHONPATH python3 -m perception_costmap.costmap_node \
    --ros-args --params-file config/perception_costmap.yaml
```

`attach_ros()` now creates the per-camera `Image` subscriptions, the lidar
`PointCloud2` subscription and the publish timer, all from the config block.
Decoding is plain numpy in `util.image_msg_to_bgr` / `util.pointcloud2_to_xyz`
rather than cv_bridge -- that keeps the subscription path in the offline test
suite (`test/test_msg_decode.py`) and keeps cv_bridge off the Jetson
dependency list. `pointcloud2_to_xyz` reads x/y/z by declared field offset,
so a real driver's x,y,z,intensity,ring cloud decodes correctly.

Verified end to end against CARLA (2026-07-31): `/perception/costmap` at
**10.005 Hz**, 200x200 cells at 0.1 m, origin (-4.0, -10.0), with free /
lethal / off-road / inflation costs all present.

Two things to know before running it:

- **`use_sim_time` must be false** unless something actually publishes
  `/clock`. `carla_feed.py` stamps with the wall clock, so leaving it true
  makes every reading look infinitely stale and the costmap stays unknown
  forever. The shipped config sets it false.
- **Frame size is pinned** by `image_width`/`image_height`, because
  `known_mask` is precomputed from it. Frames of any other size are logged
  and dropped rather than silently misprojected.

### Going blind publishes UNKNOWN, not the last thing we believed

If no camera and no lidar is fresh, `_tick` short-circuits and publishes an
all-UNKNOWN grid. Nav2 handles an unknown costmap; it cannot detect a
confidently wrong one.

This matters because `TemporalObstacleFilter` only decays confidence on
cells it *observed* this tick, and a blind tick observes none. Before the
short-circuit, killing the feed left the node publishing at 10 Hz with
**6.69% of cells still lethal, bit-identical, 40+ seconds later**. The
staleness guards were working (road dropped to 0% free immediately); the
latch was downstream of them. It survived the offline suite because every
"no fresh sensors" test there starts from a virgin node with nothing latched
-- `test_going_blind_clears_previously_latched_obstacles` establishes a
confident obstacle first, which is what makes it a real regression test.

Temporal confidence is deliberately *not* reset while blind, so recovery
from a brief dropout keeps its history rather than restarting from zero.
Obstacles that genuinely disappeared decay normally once cells are observed
again.

Verified live (2026-07-31): feed running 3.9% unknown / 4.5% free / 6.1%
lethal -> feed killed, 100.00% unknown and 0.00% lethal held across 250
messages / 25 s at a steady 10 Hz -> feed restarted, 3.9% unknown / 9.5%
free / 6.6% lethal.

## 6. Nav2 acceptance checklist

Run it instead of eyeballing it -- 300 ms and 800 ms look identical in RViz:

```bash
# with CarlaUE4.sh + tools/carla_feed.py + costmap_node already running
PYTHONPATH=. python3 tools/carla_acceptance.py
```

It brakes the ego (so the walker can't leave frame mid-measurement), spawns
a pedestrian 6 m ahead, and timestamps the costmap transitions. Measured
2026-07-31 on Town10HD_Opt:

| criterion | target | measured |
|---|---|---|
| `/perception/costmap` rate | >= 8 Hz | **10.00 Hz** |
| lethal cells on clear road ahead | 0 | **0** |
| pedestrian appears -> lethal | ~300 ms | **168 ms** (`68ms:0 168ms:4`) |
| pedestrian removed -> clear | ~500 ms | **299 ms** (`7ms:6 99ms:5 200ms:4 299ms:0`) |

### The road ahead was LETHAL until `lidar_z_min` was fixed

The first run of the above reported **44-63 lethal cells within 1 m of every
probe point on the road ahead**, and no clear region existed anywhere to
measure against. Cause: `remove_ground_plane`'s shipped `z_min = -0.3`.

Lidar points reach that filter in base_link with the mount height already
added, so road returns sit at z ~= 0 -- a *negative* z_min keeps every one of
them, and `build_cost_array` has obstacles override road, so the entire
drivable surface published as LETHAL. Measured against CARLA: of the lidar
points in the 2-16 m box ahead, **100% were road surface within +/-0.15 m of
z = 0, and all of them survived the filter**. At `z_min = 0.15`, zero
survive there while a pedestrian (metres tall) is untouched.

It looked plausible in RViz -- a busy town scene *should* have obstacles --
and the offline test for this function used z = -1.0 as its "ground" point,
a value no real lidar produces. `test_default_z_min_rejects_actual_road_returns`
now pins the default against realistic road noise instead.

The default is now `0.15` in `obstacles.py`, `costmap_node.py` and
`config/perception_costmap.yaml`. Curbs (~0.12 m) are below it and will not
be seen by lidar; they come from the camera road mask.

### Nav2 mirror check

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
ros2 run nav2_costmap_2d nav2_costmap_2d \
    --ros-args -r __ns:=/local_costmap -r __node:=local_costmap \
    --params-file config/nav2_costmap_params.yaml \
    -p always_send_full_costmap:=true
ros2 lifecycle set /local_costmap/local_costmap configure
ros2 lifecycle set /local_costmap/local_costmap activate
```

Verified: `StaticLayer: Resizing static layer to 200 X 200 at 0.100000 m/pix`
on activation, and the same pedestrian test read 0 -> 5 -> 0 lethal cells at
6 m in **both** `/perception/costmap` and `/local_costmap/costmap`, with 112
additional cells at cost 99 around it from Nav2's own `inflation_layer`.

Three things that cost time here:

- **`always_send_full_costmap: true` is needed to compare them.** By default
  Nav2 publishes one full map and then incremental
  `/local_costmap/costmap_updates`, so `/local_costmap/costmap` looks dead
  (`ros2 topic hz` times out) while the costmap is in fact updating fine.
- **TF is required** -- nothing in this package or `carla_feed.py` publishes
  `map -> odom -> base_link`. On the real car that comes from odometry; in
  sim the static publishers above are enough to run the check.
- **`obstacle_layer` had no input.** `config/nav2_costmap_params.yaml` points
  it at `/perception/obstacle_points`, which nothing publishes -- the node
  only publishes the OccupancyGrid. The mirror above works entirely through
  `static_layer`. Publishing the filtered obstacle cloud is still to do; the
  practical effect of its absence is that Nav2 gets no raytrace *clearing* of
  its own, and relies on this package's temporal filter for that.

## 7. Jetson-specific notes (only if targeting real hardware)

- Never `pip install torch` on the Jetson -- it pulls a CPU wheel. Use
  NVIDIA's Jetson-matched wheel, then `pip install ultralytics --no-deps`.
- `tools/export_trt.py` must run ON the Jetson -- a `.engine` built on the
  CARLA/dev box will not load on different silicon.
- `python3 tools/bench_perception.py --frames 50 --yolo-weights
  yolov8n.engine` -- re-run this on-device; the numbers from this sandbox
  (2 CPU cores, no GPU, ~300 Hz on the classical/HSV path) do not transfer.
