# Working notes for AI assistants

Read this before trusting anything else in the repo. It says what is real,
what is stub, and where to start.

## What this project is

Sim-to-real autonomous vehicle pipeline: validate perception + Nav2 in CARLA
with the same 3-camera layout as the real car, then deploy the identical ROS2
stack to the car's onboard computer (an NVIDIA Jetson AGX Orin 64 GB, JetPack 6.1 /
Ubuntu 22.04 / native ROS2 Humble). CARLA is x86-only and never runs on the
Jetson — only the sensor source changes between sim and car.

## Where the real work is

The active, tested, reviewed work is the **`perception_costmap`** package on
the **`feature/alexander`** branch:

- `ros2_ws/src/perception_costmap/` — camera+lidar perception publishing a
  Nav2-compatible costmap. Multi-camera BEV fusion, HSV or TwinLiteNet+ road
  segmentation, classical or YOLOv8 obstacles (footprint-strip projection,
  .pt or TensorRT .engine), temporal confidence filter, sensor-data QoS,
  staleness guards. 39 offline tests.
- `ros2_ws/src/perception_costmap/README.md` — build/run/Nav2 wiring.
- `ros2_ws/src/perception_costmap/DESIGN.md` — architecture + dataflow.
- `ros2_ws/src/perception_costmap/DEPLOY.md` — Jetson bring-up checklist.
- `ros2_ws/src/perception_costmap/tools/` — ipm_overlay (calibration check),
  carla_feed (CARLA 0.9.16 → ROS2 topics, runs on the x86 sim box only),
  eval_road_iou (segmenter accuracy vs CARLA semantic ground truth),
  export_trt (run ON the Jetson), bench_perception (per-stage timing).
- `docs/plans/2026-07-01-perception-v2-sim-to-real.md` — the implementation
  plan this was built from (all 10 tasks complete).
- `driving_seg/` (repo root) — multi-model area-highlighting segmentation
  (people/vehicles/signs/road/lanes/cones/white lines, no bounding boxes);
  standalone package with its own README/PROMPT.md, trained cone model
  committed. Real, tested work (2026-07-07).
- `perception/` (repo root) — Adam Castillo's original prototype scripts the
  package was factored from. Keep author credits intact.

## What is NOT real (do not build on these)

- `ros2_ws/src/collision_guard`, `route_planner`, `sdc_common` — empty stubs.
- `ros2_ws/src/controller` — setup.py declares nodes whose files don't exist.
- `ros2_ws/src/sdc_bringup/launch/sdc.launch.py` — broken (foreign hardcoded
  path, references nodes that don't exist). Use
  `perception_costmap/launch/perception.launch.py` instead.
- CONTRIBUTION_GUIDE.md's stack description (CARLA 0.10 / controller /
  planner) is partly aspirational — trust the perception_costmap docs.
  The root README was rewritten truthful (2026-07-07).

## How to verify a checkout (no ROS needed)

    cd ros2_ws/src/perception_costmap
    PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q     # 39 passed
    python3 tools/bench_perception.py --frames 20          # stage table

With ROS2 (Humble target; Jazzy works for build/import):

    cd ros2_ws && colcon build --packages-select perception_costmap
    source install/setup.bash
    ros2 launch perception_costmap perception.launch.py

## Conventions that bind changes

- Python 3.8-compatible syntax (Jetson floor). No `match`, no `X | Y` unions.
- torch / ultralytics / carla / ROS message types are optional, lazily
  imported. The pytest suite must pass with only numpy + opencv installed.
- Core modules (`segmentation`, `obstacles`, `bev`, `occupancy`, `temporal`,
  `carla_convert`, `util`) stay ROS-free; only `costmap_node.py` imports rclpy.
- Grid math: REP-103 (+x forward, +y left); OccupancyGrid row-major,
  costs -1 unknown / 0 free / 100 lethal. CARLA is left-handed (y right) —
  conversions live in `carla_convert.py`, don't re-derive them.
- Commit style: `perception_costmap: <what>`. No AI co-author trailers.
- Team: alexander (arassal) leads; jchy05, AdamCastillo07, Ad-Tap are mentees
  with their own feature branches. Don't rewrite their branches.

## What still needs a human or hardware

- Per-camera IPM calibration: YAML `ipm_*` values are placeholders — use
  `tools/ipm_overlay.py` against a real frame (procedure in package README).
- CARLA smoke test + IoU accuracy table: run on the x86 sim box with CARLA
  0.9.16 (`tools/carla_feed.py`, then `tools/eval_road_iou.py`).
- Jetson: TensorRT export + benchmark on-device, real camera/lidar drivers —
  follow DEPLOY.md top to bottom. Machine access details are NOT in this repo;
  ask the team.
