# perception_costmap_v2

A from-scratch reimplementation of the camera+lidar -> Nav2 costmap
pipeline from `arassal/carla-nav2-avl` (`perception_costmap` package,
`feature/alexander` branch), built and verified in an environment with
**no GPU and no CARLA installed**. See `DEPLOY.md` for what to do once you
move this to a machine that has both.

## What's actually verified here (no GPU, no CARLA)

- **68/68 pytest tests pass**, covering every core module with only
  numpy + opencv installed (`test/`).
- **`tools/synthetic_demo.py`** runs the full pipeline end-to-end --
  segmentation -> BEV warp -> lidar binning -> temporal filter ->
  cost fusion -- against synthetic (not CARLA) camera/lidar data, and
  renders `demo_output/synthetic_demo.png` proving:
  - the road segments correctly and warps to the right BEV cells
  - a pedestrian and a cone each produce a lethal cell with a distinct,
    class-appropriate inflation halo
  - the temporal filter requires two hits to mark an obstacle lethal, and
    survives one missed frame (occlusion) without clearing it
- **`tools/bench_perception.py`** measures ~300 Hz on the classical/HSV
  path on 2 CPU cores with no GPU -- comfortably above the 8 Hz Nav2
  target even here; real numbers on your hardware will differ (that's
  the point of re-running it there, see DEPLOY.md).

## What is NOT verified here (needs your GPU/CARLA machine)

- `tools/carla_feed.py` -- real CARLA connection, sensor spawning,
  ROS2 publishing. Needs `carla==0.9.16` + a running `CarlaUE4.sh` +
  a GPU. Lazy-imports both `carla` and `rclpy` so importing this file
  elsewhere never requires either.
- `tools/eval_road_iou.py`'s live half -- scoring segmenters against
  real CARLA semantic ground truth (the offline IoU math itself
  (`carla_convert.mask_iou`) is unit-tested here).
- `costmap_node.py`'s ROS2 wiring (`attach_ros`) -- the fusion logic
  (`_tick`) is fully tested without rclpy; the subscription plumbing
  needs a live ROS2 graph to exercise.
- `tools/export_trt.py` / TensorRT engines -- hardware-specific, must
  run on the target device.

## A real bug this exercise found

Building the synthetic demo surfaced an actual geometry bug: warping a
camera-derived obstacle mask through the IPM homography without clipping
it to the camera's own observed footprint (`bev.bev_known_mask`) can
plant a spurious obstacle cell via a "mirror cell" -- a destination grid
cell whose inverse-mapped source ray has near-zero/negative projective
depth still gets sampled from real image content. Fixed in
`obstacles.camera_obstacle_mask_to_grid(..., known_mask=...)`, wired
through `costmap_node.py`, and covered by
`test_footprint.py::test_camera_obstacle_mask_to_grid_clips_to_known_mask`.
Lidar-derived obstacles are deliberately left unclipped (metric points,
not warped image content -- a spurious real-sensor obstacle is the safe
direction; see `occupancy.py`'s own docstring for that convention).

## Layout

```
perception_costmap/
  occupancy.py       GridSpec, build_cost_array, inflate_costs, infill_unknown
  bev.py              IPM homography (points or camera-extrinsics), warp, known-footprint
  temporal.py         per-cell obstacle confidence filter
  obstacles.py         lidar binning, footprint-strip rasterizer, classical + YOLO detectors
  segmentation.py     HSV / TwinLiteNet+ segmenters behind one factory interface
  carla_convert.py    pure CARLA<->REP-103 conversions (left-handed fix, semantic tags)
  costmap_node.py     the only module that imports rclpy; _tick() is pure and testable
  util.py             stamp/staleness helpers
test/                 68 tests, numpy+opencv only, no ROS/CARLA/torch required
tools/
  synthetic_demo.py    CARLA-free end-to-end demo (this environment)
  ipm_overlay.py       calibration check -- draws the metric grid back onto a frame
  carla_feed.py         real CARLA -> ROS2 topics (GPU box only)
  eval_road_iou.py     segmenter accuracy vs CARLA semantic ground truth
  export_trt.py        YOLOv8 -> TensorRT (run ON the target device)
  bench_perception.py  per-stage timing (re-run on every target machine)
config/
  perception_costmap.yaml    node parameters, camera list, cost-shaping knobs
  nav2_costmap_params.yaml   Nav2 layer wiring (static_layer + obstacle_layer + inflation)
demo_output/
  synthetic_demo.png   output of tools/synthetic_demo.py
DEPLOY.md              step-by-step: what to do on your CARLA/GPU machine
```

## Quick start (anywhere -- no GPU needed)

```bash
pip install -r requirements.txt --break-system-packages   # or a venv

PYTHONPATH=. python3 -m pytest test -q                    # 68 passed
PYTHONPATH=. python3 tools/synthetic_demo.py               # writes demo_output/synthetic_demo.png
PYTHONPATH=. python3 tools/bench_perception.py --frames 20
```

## Next step

Unzip this next to your real CARLA checkout and follow `DEPLOY.md` --
it walks through connecting real CARLA, calibrating cameras with
`ipm_overlay.py`, measuring real segmenter IoU, and running the ROS2 node.
