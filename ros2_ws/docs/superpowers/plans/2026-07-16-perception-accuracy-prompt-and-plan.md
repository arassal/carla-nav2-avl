# Perception Accuracy — Context Brief & Action Plan

**Date:** 2026-07-16
**Rollback point:** git tag `stable-2026-07-16` on both `carla-nav2-avl` (feature/alexander) and `IGVC` (main), plus plain directory copies at `/home/dinosaur/backups/2026-07-16/`. Everything below is additive/incremental work on top of that known-good state — if anything goes wrong, `git checkout stable-2026-07-16` or restore from the backup copies.

---

## Part 1 — Context Brief (read this first; self-contained, no prior conversation needed)

### The vehicle
IGVC competition ground vehicle ("dinosaur"), NVIDIA Jetson Orin Nano, ROS 2 Humble. **Top speed ~0.7-1.5 m/s (walking pace) by deliberate design** — this is not a high-speed platform, and every latency/performance judgment below is calibrated to that fact, not highway speeds.

### Sensors
- 3x ZED X stereo cameras (front/left/right), each publishing RGB, depth (`NEURAL_LIGHT` mode, always on), and point clouds.
- Velodyne VLP-16 lidar — **currently non-functional**. `eno1` (its Ethernet NIC) shows `NO-CARRIER`; zero packets reach the driver even with the interface forced up. This is a physical-layer problem (cable/connector/power), not software, and cannot be fixed remotely. Matches a pattern of vibration-loosened connectors seen elsewhere on this vehicle (the Jetson's own DC power connector has caused repeated hard power-cuts).
- Xsens IMU, RTK GNSS — both healthy.

### Perception pipeline (camera-only right now, since lidar is down)
`perception_costmap` package (`carla-nav2-avl` repo, `feature/alexander` branch):
- **Road detection**: TwinLiteNet (nano, TensorRT) semantic segmentation per camera, minus painted white lines.
- **Obstacle detection**: YOLOv8n (COCO classes, grouped into `person`/`vehicle`) + a dedicated cone detector (color-gated + detector boxes).
- **Projection**: each camera's detections are warped into a shared 200x200 (20m x 20m, 0.1m/cell) bird's-eye-view grid via a calibrated homography (`bev.homography_from_extrinsics`), using **person-walk-measured** pitch/yaw/xyz per camera (not the URDF/TF mount angles — those are known to be stale, see Known Issues below).
- **Cost model** (`occupancy.build_cost_array`): graded, not flat —
  - Free road: 0. Off-road (grass/curb/wall): 97 (near-lethal, not a hard 100, so a segmentation dropout can't strand the vehicle with zero legal cells).
  - Road-edge ramp: cost rises smoothly from 0 to 97 as you approach the road boundary from inside the lane (1.5m radius, so the planner has a real gradient to centre in, not a cliff).
  - Per-class obstacle halos: person 2.5m radius / 1.5 decay (wide, slow falloff — keep well clear), vehicle 1.0m / 5.0 (tight — passing close to a car is normal), cone 0.6m / 5.0, unclassified blob 0.8m / 4.0 (legacy default).
  - Blind-spot infill: unobserved cells (camera seams, the ~106° rear blind wedge — no rear camera exists) are Telea-inpainted from surrounding observed cost, decaying to a neutral "unknown" prior (25) with distance from the nearest real observation — so a seam between two cameras both seeing road reads as road, not a false wall.
  - FOV-edge trim: each camera's coverage mask is eroded 0.4m inward at its border, because segmentation is unreliable in the outermost pixels — this was previously causing phantom off-road bands hugging every camera seam.
- **Nav2 integration**: `/perception/costmap` is ingested as a `StaticLayer` at the base of both the local and global Nav2 costmaps (`trinary_costmap: false` set at the correct costmap-node level — a per-layer `perception_layer.trinary_costmap` is silently ignored by Nav2 Humble, a real bug that was caught and fixed). Lidar's `stvl_layer` sits above it in the plugin stack but currently contributes nothing (see lidar status above).

### What's validated, and how (not just claimed)
- **Live NavigateToPose test**: a real ~6m goal, MPPI planning through the fused costmap, `SUCCEEDED` at the 0.7 m/s cap. A person in the path produced measurable (0.22m) lateral avoidance shading — real, but modest; see Known Issues.
- **Shadow robustness**: synthetic shadows injected into real recorded frames, run through the actual TensorRT segmentation engine. Realistic shadows cost ~0% road detection; only extreme (near-dusk) darkness shows measurable (~18%) loss, in scattered fringe pixels, not region flips.
- **depth_obstacle_node**: a new, standalone node (deliberately **not** wired into the main costmap yet — kept independent per explicit decision, to prove out alone first) that converts each camera's point cloud into a robot-frame obstacle grid, using the same calibrated extrinsics as the BEV code (not TF — see Known Issues). Two real bugs were caught and fixed before this was trusted:
  1. TF's static mount angles disagree with the calibrated extrinsics (proven: left camera TF says pitch=0°/yaw=90°, calibrated config says pitch=18°/yaw=94°). Using TF produced physically implausible results (side cameras keeping ~90% of points in the obstacle height band vs front's ~50%); switching to calibrated extrinsics gave a consistent 35-46% across all three.
  2. Running it as a separate OS process did **not** by itself protect the main costmap's tick rate — measured live, the main costmap dropped from 5.85→2.8 Hz with the depth node running unthrottled (Jetson's shared memory bandwidth is the likely bottleneck, not CPU core count — 12 cores are nominally free). Fixed with 8x point-cloud downsampling; two independent 25-second sustained checks confirmed the main costmap holds ~6.4-6.5 Hz with the depth node running and still producing real, non-trivial obstacle data.
  - **Everything above was validated stationary.** Real drive-time CPU load (joystick polling, actuator serial I/O, GPS/NTRIP, motion itself) has never been tested. Treat the depth pipeline as "proven safe at rest," not "proven safe while driving," until a real test drive confirms it.
- **SVO recording**: built and live-tested (recorded all 3 cameras simultaneously under full stack load, zero measured rate impact; played a recording back through the stock `zed_camera.launch.py` unmodified, RGB+depth+IMU all reproduced correctly). **This is purely an offline tool** — it does not affect real-time detection in any way, and does not need to be "on" for anything else here to work. On-demand only by design (auto-starting would fill the 93%-full disk).

### Known issues (in priority order for the plan below)
1. **Right camera's extrinsics are stale.** The right camera's mount was physically moved (to reduce the rear blind spot) after the 2026-07-09 person-walk calibration, and has **never been re-measured**. `perception_dinosaur.yaml` still has the pre-move pitch=20°/yaw=-86°. This is wrong geometry feeding the **live, currently-running** costmap right now — not a hypothetical, not limited to the new depth work.
2. **Lidar is physically disconnected.** Needs hands-on cable/connector inspection, not software.
3. **depth_obstacle_node is not integrated anywhere.** It runs, it's correct, it's fast enough — but nothing consumes its output yet. A deliberate, cautious integration path is needed (see Plan).
4. **Person-avoidance margin may be too thin.** One real test showed only 0.22m of lateral shading around a detected person. Could be the `person_radius`/`person_scaling` cost tuning, could be MPPI's own critic weights not respecting the costmap gradient as strongly as expected — undiagnosed, needs investigation before trusting it around people at closer range.
5. **Disk is at 93% full** (`bags/` alone is 686GB). Not blocking anything today, but will start causing real failures (failed recordings, failed writes) if it fills further. Not perception-accuracy-related, but worth keeping on the radar since SVO/bag recording work will make it worse.

---

## Part 2 — Action Plan

Ordered by (impact × how cheap it is to do), not strictly by dependency — items 1-2 can happen independently of each other.

### 1. Recalibrate the right camera (highest priority — fixes a real, live bug)
Repeat the same person-walk method used 2026-07-09 for the other two cameras, for the right camera only, at its new mount position: an operator walks at known bearings in the right camera's view while YOLO detects them; cross-reference against a physical landmark (e.g. a brick course / straight edge) for the vanishing-point check, same as before. Update `perception_dinosaur.yaml`'s `right.cam_pitch_deg` / `right.cam_yaw_deg`, and the matching constants in `deploy/depth_obstacle_node.py`, `deploy/costmap_rgb_node.py`, `tools/viz_node.py` (these currently duplicate the extrinsics as local constants rather than reading the YAML — worth eventually consolidating to one source of truth, but not blocking). Re-run the shadow-robustness and a stationary RViz sanity check afterward to confirm the right camera's coverage/road-edge geometry looks sane again.

**Requires physical access to the vehicle** (a person walking in front of it). Cannot be done remotely.

### 2. Get the lidar working (high priority, physical work)
Inspect the Ethernet cable/connector between the Jetson (`eno1`) and the Velodyne interconnect box; confirm the lidar has power. Once link is confirmed (`ethtool eno1` shows "Link detected: yes", and `ros2 topic hz /velodyne_points` shows data), the existing `stvl_layer` in both Nav2 costmaps will start contributing automatically — no code changes needed, it's already wired in and just waiting for data.

**Requires physical access to the vehicle.**

### 3. Decide and implement how depth_obstacle_node's output gets used (medium priority, needs a real drive to fully validate)
Recommended phased approach, most cautious first:
- **Phase A (log-only cross-check):** subscribe to `/perception/depth_obstacle_grid` from a small monitoring script; log/alert whenever it flags a cell the main RGB-based costmap does *not* already flag as elevated cost. This tells you how often depth is catching something RGB misses, with zero risk — it doesn't touch driving behavior at all. Do this across a few real test-drive sessions before trusting the signal.
- **Phase B (soft merge):** once Phase A shows the signal is reliable, merge depth-derived obstacles into `costmap_node`'s existing `obstacle_layers` dict as a new `depth` class (own radius/scaling, tuned conservatively — probably closer to `generic` than to `person`, since depth alone can't tell you what kind of object it found), OR feed `/perception/depth_obstacle_grid` as its own additional Nav2 `StaticLayer`, same pattern as `/perception/costmap` already uses.
- **Do not skip Phase A.** The depth pipeline has only been validated stationary; a real drive is the first legitimate test of whether it holds up under actual load, and Phase A gets you that data without betting driving behavior on it first.

### 4. Investigate the thin person-avoidance margin (medium priority)

**UPDATE 2026-07-16 -- investigated; coherent root cause found (needs a drive to confirm).**
Nothing is attenuating the halo -- the graded field reaches Nav2 correctly
(verified: the 0-100 OccupancyGrid is scaled into Nav2's internal 0-254 range,
and Nav2's own InflationLayer combines with max(), so it cannot reduce our
2.5m person halo). The likely cause is a *planning* interaction, not a
perception bug:

- **The person halo is deliberately non-lethal beyond its core.** Core = 100
  (-> 254 = lethal, avoided), but at 0.5m out it is only 47 (-> ~119 in
  Nav2 scale) = "traversable with penalty".
- **The global planner is NavFn** (Dijkstra, point-robot, allow_unknown true).
  A minimum-cost planner will happily route *through* a cost-119 band if
  detouring around costs more total path length. So the global path likely
  clips close past the person rather than routing wide around them.
- **MPPI is then tuned to follow that path tightly.** PathAlignCritic
  cost_weight 16.0 vs CostCritic cost_weight 6.0 -- path-following is ~3x
  stronger than obstacle-cost avoidance. That 8.0 -> 16.0 raise was
  deliberate (2026-05-30, to stop MPPI "looping around every halo" and
  wandering 30m off a 15m route), and its own config comment warns:
  "Tune down if it gets too rigid to dodge a real obstacle." That is
  precisely the symptom observed.
- **Nav2 own inflation is intentionally thin** (inflation_radius 0.4 ~=
  robot half-width, so NavFn threads gaps accurately). Its comment
  explicitly accepts the tradeoff: "band 0.4-0.28=0.12 m = thin MPPI
  gradient (early-steer is weak) -- that is why the monitor is needed for
  the slow-down", referring to nav2_collision_monitor, which is marked
  **(pending)** and does not exist yet. Note our perception layer wide
  graded halo actually *solves* the very problem that comment laments --
  but only if the critic balance lets it.

**Recommended experiment (requires a test drive), change ONE variable at a time:**
1. First just observe: with a person standing off to the side of a straight
   path, does the *global* path (/plan) route around them, or straight
   through their halo? That single observation discriminates "NavFn plans
   through" from "MPPI ignores the plan" and tells you which knob matters.
2. If the global path goes through: raise person_radius and/or lower
   person_scaling in perception_dinosaur.yaml so the halo stays costly
   further out, making the detour cheaper than the pass-through.
3. If the global path routes around but MPPI cuts the corner anyway: lower
   PathAlignCritic (16.0 -> 12.0) and/or raise CostCritic (6.0 -> 8.0), in
   small steps -- 16.0 exists to fix a real past failure (wandering off
   plan), so do not undo it wholesale.

Original notes below.
Two independent things to check, ideally with an actual test drive (a stationary RViz costmap inspection can only get you partway):
- Re-verify the cost values actually reaching Nav2 near a person match what `occupancy.py` computes in isolation (47 @ 0.5m, 22 @ 1m, 4 @ 2m) — confirm nothing is attenuating the halo between the perception costmap and what MPPI actually sees (e.g. Nav2's own `InflationLayer` interacting oddly with an already-graded StaticLayer costmap is a real thing to check for).
- Review MPPI's critic weights (`nav2_params_humble.yaml` / `nav2_params_igvc_autonav.yaml`) — specifically whether the obstacle/cost critic is weighted strongly enough relative to the goal-tracking critic that a graded (not binary) cost field actually produces a wide swerve rather than a minimal deflection.
- If both check out and the margin is still thin, consider raising `person_radius` and/or lowering `person_scaling` (slower decay = cost stays elevated further out) in `perception_dinosaur.yaml`.

### 5. Use SVO to build a real regression-test habit (lower priority, pure infrastructure investment)
Now that recording is proven safe and cheap: record a short SVO session covering a real test drive (once one happens), and build out `shadow_robustness_test.py`'s pattern into a small library of "does detection still work" checks that can be re-run against saved SVO clips after any model swap, threshold change, or camera recalibration — catching regressions before they reach the live vehicle, the same way the shadow test already does for lighting.

### 6. Disk space (housekeeping, not accuracy, but do it before it becomes a real problem)
`bags/` is 686GB and the disk is at 93%. Not urgent today, but should be addressed (archive/thin old bags, or move them off-device) before the next round of SVO recordings or field-test bag captures pushes it over the edge.

---

## What this plan deliberately does NOT include
- Retraining or swapping the segmentation/detection models (TwinLiteNet-nano, YOLOv8n, the cone detector). These are real, valid future levers for accuracy, but are a much larger ML-engineering effort (data collection, training infrastructure, evaluation) than anything above, and nothing today's testing suggests they're the current bottleneck — the calibration and lidar issues are cheaper, higher-confidence wins first.
- Any change aimed at highway-speed operation. This vehicle tops out around 1.5 m/s; nothing here should be over-engineered for a use case that doesn't apply.
