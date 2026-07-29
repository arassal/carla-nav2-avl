# FINALIZE — the prompt that takes this project to done

Paste this file (or point a session at it) to drive the remaining work.
Execute phases IN ORDER; each has acceptance criteria — do not advance on
red. Everything below is verified fact as of 2026-07-07 unless marked open.

---

## Mission

One perception system on the car: accurate computer vision (people,
vehicles, signs, road, lanes, **cones**, **white course lines** — as area
masks, never bounding boxes) fused into `/perception/costmap` +
`/perception/obstacle_points`, consumed by ROS2 Nav2, running hands-off at
≥ 8 Hz on the Jetson from power-on to autonomous driving. No babysitting,
no silent failures.

## Where we are (verified)

- `ros2_ws/src/perception_costmap`: 3× ZED X + velodyne → fused costmap.
  39 offline tests green. Runs on the car ~4–5 Hz (TwinLiteNet 74 ms is
  the bottleneck). Mirror-projection + degenerate-homography + velodyne
  dtype crashes all fixed and regression-tested.
- Nav2 verified consuming both perception topics (static + obstacle +
  inflation layers) with EKF providing `odom`. Lifecycle bringup manual.
- `driving_seg/`: standalone 8-class highlighter, 65 FPS on the 5090.
  Cone stage is a hybrid — pretrained detector (`models/cone_det.pt`,
  catches striped cones) + in-box color/hull area masks; white lines from
  our fine-tuned `models/course.pt` (weak: 15 training images).
- Car ops: boot autostart (`percept-stack.service`, enabled), camera
  watchdogs, GMSL recovery script, live dashboard :8090, joystick :8000.
- **The gap that matters most: cones and white lines do not reach the
  costmap yet.** They exist only in driving_seg. For IGVC they ARE the
  course — closing this is Phase 1.

## Phase 1 — cones + white lines INTO the costmap (the core work)

Port the two course classes into `perception_costmap` (keep core modules
ROS-free; follow the package's existing conventions):

1. `perception_costmap/obstacles.py`: add a `ConeDetector` modeled on
   `YoloObstacleDetector` but loading `cone_det.pt` (detector → in-box
   color/hull area mask; port `_box_to_cone_mask` from
   `driving_seg/models/course.py`). Config keys: `cone_weights`,
   `cone_conf` (default 0.35), `use_cones` (default true).
2. `perception_costmap/segmentation.py`: white_line as a NEGATIVE road
   mask — subtract line pixels from the drivable mask so painted
   boundaries become off-road (lethal) in `build_cost_array`. Source:
   either `course.pt` masks or the classical white-on-grass gate from
   `driving_seg/tools/build_course_dataset.py:label_white_lines` (start
   classical — it's free and runs in 2 ms; swap learned later).
3. Cones → `obst_grid` via the same per-camera warp path (`& cam.known` —
   don't reintroduce the mirror bug). Add offline tests: synthetic orange
   triangle → lethal cells at the right grid position (asymmetric fixture:
   cone left of center must land left).
4. Export `cone_det.pt` to TensorRT ON the Jetson (`yolo export
   format=engine half=True`), wire the `.engine` path in the car config.
   Budget: cone stage ≤ 15 ms/camera on Orin.

Accept: person AND cone placed in front of the car both become lethal
cells within 300 ms (watch `/viz/costmap_render`); 39+ tests green; rate
not below current.

## Phase 2 — speed to ≥ 8 Hz on the car

1. Reboot into MAXN (`sudo nvpmodel -m 0`, needs reboot — boot service
   brings the stack back) + `sudo jetson_clocks`. Re-bench.
2. TwinLiteNet → TensorRT (74 ms → target ≤ 15 ms): export its ONNX at
   384×640, build FP16 engine with trtexec, add an `.engine` path to
   `TwinLiteSegmenter` mirroring how YOLO weights accept engines. If it
   resists, fallback: run seg on front camera only + classical gates on
   sides (config already supports per-camera choices via code change).
3. Consider running the 3 cameras through one batched inference instead
   of serial (biggest architectural win; measure first).

Accept: `ros2 topic hz /perception/costmap` ≥ 8 Hz sustained with all
three cameras + lidar + cones, on battery power.

## Phase 3 — geometry truth (IPM calibration)

Tape measure + `tools/ipm_overlay.py` per camera (procedure in the package
README). Replace the URDF-derived numbers in
`config/perception_dinosaur.yaml` with measured `ipm_image_pts` /
`ipm_world_pts` (points mode) per camera.
Accept: object at measured (x, y) lands within 1 cell (0.1 m) on
`/perception/costmap` for front, left, right.

## Phase 4 — Nav2 for real (not the standalone probe)

Use the team's `~/IGVC` `navigation.launch.py` (Nav2 servers + lifecycle +
route server). Integration rules learned the hard way:
- ONE actuator_node only — it's already running under `avros-webui.service`;
  launch navigation WITHOUT its actuator or stop the webui service first.
  Two instances fight over the Teensy serial port.
- EKF must be up before velodyne/Nav2 (odom frame gates both). The percept
  stack's `ekf` window provides it; don't double-run it.
- Drop `config/nav2_costmap_params.yaml` layers into their nav2 params
  (static_layer ← `/perception/costmap`, obstacle_layer ←
  `/perception/obstacle_points`; keep obstacles unclipped for lidar).
Accept: with the car on stands or in open space — send a Nav2 goal 5 m
ahead via RViz/foxglove; local costmap shows perception data; planner
produces a path that AVOIDS a cone placed in the corridor; `/cmd_vel`
output sane (≤ configured caps). Then a supervised live run.

## Phase 5 — field acceptance (competition rehearsal)

On grass with painted lines + cones (the real course setup):
1. `bash ~/carla-nav2-avl/ros2_ws/src/perception_costmap/deploy/../..` —
   power-cycle test first: cold boot → everything up unaided → all-green
   health (write car_health.sh from TESTPLAN if not present).
2. White lines: drive the joystick along the course; `/viz/costmap_render`
   must show line cells as lethal boundary consistently ≥ 5 m ahead.
   If weak: photograph 50–100 course shots, retrain
   (`driving_seg/docs/CONE_DETECTION.md` recipe, ~30 min), re-export.
3. Cone slalom: 6 cones, 3 m spacing; teleop through while recording
   (`ros2 bag record /perception/costmap /viz/fused_bev`); every cone
   lethal at ≥ 5 m range, zero phantom cones.
4. Person test: step in front → lethal ≤ 300 ms, clears ≤ 500 ms.
5. Autonomous slalom with a safety driver on the e-stop.

## Standing rules (do not relearn these)

- Env on the car, always: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`,
  `CYCLONEDDS_URI=file:///home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml`.
- tmux is on a dedicated socket: `tmux -L percept attach`.
- Cameras wedge on daemon restarts → `deploy/clean_camera_restart.sh`,
  never restart `zed_x_daemon` under live wrappers.
- Nested quoting kills PYTHONPATH → runner scripts (`deploy/run_viz.sh`
  pattern), never inline multi-layer tmux/bash strings.
- Jetson pip: `https://pypi.jetson-ai-lab.io/jp6/cu126`, torch 2.8.0,
  `numpy<2`. Engines are built ON the device they run on.
- Verify visual/geometry changes with ASYMMETRIC fixtures (mirror-bug
  class); verify packaging with a FRESH CLONE (three bugs caught that way).
- Repo conventions: `CLAUDE.md` binds — Python 3.8 syntax, ROS-free core
  modules, `perception_costmap: <what>` commit style, no AI trailers.

## Definition of DONE for the whole project

Cold power-on → (no human input) → cameras, EKF, lidar, costmap, Nav2 up →
costmap ≥ 8 Hz showing road free / cones + lines + people lethal, geometry
calibrated to ±0.1 m → Nav2 accepts a goal and drives the cone slalom on
grass without touching a cone or crossing a line, e-stop verified, three
runs in a row, with the live dashboard streaming it at :8090.
