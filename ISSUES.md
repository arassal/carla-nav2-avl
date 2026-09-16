# ISSUES — bugs and open problems on this branch

Working log of what is broken, found while setting up a **fresh clone** of
`copy` on a laptop (Ubuntu, ROS 2 Jazzy, no GPU sensors attached) on
2026-09-10. `FINALIZE.md`'s standing rules say to "verify packaging with a
FRESH CLONE (three bugs caught that way)" — B1 below is a fourth.

Status key: **OPEN** / **IN PROGRESS** / **FIXED** (commit) / **WONTFIX** (why).

Evidence for each entry goes in `logs/` — see `logs/README.md`.

---

## Blockers

### B1 — `costmap_node.py` imports four modules that do not exist — FIXED by Alexander in `0fedca3`

`costmap_node.py` imported `detection_schedule`, `inference_worker`,
`sample_buffer` and `health`, which were not committed, so a fresh clone of
`copy` could not import the node and `perception.launch.py` could not start.
The offline suite stayed green because no test imports `costmap_node`.

They existed untracked on Alexander's machine (`0e16bc3` was a
`git commit -a`, which skips new files). He committed them, with tests, in
`0fedca3` (2026-09-14). Verified on this branch: the node imports under ROS 2
Jazzy.

### B2 — autodrive click always rejected: the Nav2 cloud bridge cannot import — FIXED by `0fedca3`

**This was the reported "click a point, says Nav2 bridge not connected" bug.**

`deploy/costmap_to_cloud.py:31` imported `perception_costmap.costmap_cloud`,
which was also uncommitted. The bridge died on import, so
`/perception/costmap_cloud` was never published, so `campus_navigator.py`'s
preflight (`:296-297`) saw `_cloud_t == 0.0` and refused every destination
click with *"computer-vision Nav2 bridge is stale."*
`auto_drive.launch.py:137-138` respawns it every 2 s, so it crash-looped
invisibly rather than failing loudly.

`0fedca3` commits `costmap_cloud.py`. Verified on this branch with Alexander's
module: `deploy/costmap_to_cloud.py` fed by `tools/fake_costmap_publisher.py`
publishes `/perception/costmap_cloud` at 9.97 Hz, 401 endpoints per cloud.
(An earlier version of this branch carried a from-scratch `costmap_cloud.py`;
it was dropped in favour of his, which has the same signature.)

Full trace: `logs/results/2026-09-10_autodrive-click-rejection-rootcause.md`.

### B3 — `eval_road_iou.py` imports a module that does not exist — FIXED by `0fedca3`

`tools/eval_road_iou.py:23` imports `perception_costmap.evaluation`, which was
uncommitted. Committed in `0fedca3` with `test_evaluation.py`.

---

## Code bugs

### C1 — `GridSpec.world_to_cell` truncates toward zero, admitting out-of-grid points — FIXED (this branch)

`ros2_ws/src/perception_costmap/perception_costmap/occupancy.py:63-69`

```python
col = int((x - self.x_min) / self.resolution)
row = int((y - self.y_min) / self.resolution)
```

`int()` truncates toward zero, so a coordinate up to one full cell *below*
`x_min` / `y_min` maps to index `0` and passes the `0 <= col` bounds check
instead of returning `None`. Only the lower edge is affected; the upper edge
is correct.

Observed with the default grid (`x_min=-4.0`, `y_min=-10.0`, `res=0.1`):

| input (m) | returns | should return |
|---|---|---|
| `(-4.05, 0.0)`  | `(0, 100)` | `None` |
| `(-4.099, 0.0)` | `(0, 100)` | `None` |
| `(0.0, -10.05)` | `(40, 0)`  | `None` |

Fix is `int(np.floor(...))`. **Applied**, plus two regression tests: one
pinning the lower-edge rejection, and one asserting `world_to_cell` and
`points_to_grid_mask` bin identically — the drift that let them disagree for a
year is exactly what a shared cross-check prevents.

This is the **same bug the team already fixed once**: `points_to_grid_mask`
uses `np.floor` (`obstacles.py:215-216`) specifically to avoid it, and
`test_geometry.py:116-123` (`test_points_just_below_grid_min_are_dropped`)
pins that behaviour with the comment *"the old world_to_cell-based loop
truncated toward zero, wrongly binning points in (min - resolution, min) into
border cells; floor drops them."* The caller was fixed; the method it was
named after was not. `world_to_cell` is still live in `test_cheirality.py:34`
and `test_detect.py:49`.

Low blast radius today (no production path calls it), but it is a loaded gun
for anyone who reaches for the obvious-looking helper.

### C2 — TwinLiteNet PyTorch path feeds BGR, TensorRT path feeds RGB — OPEN

`ros2_ws/src/perception_costmap/perception_costmap/segmentation.py`

Two backends for the same network disagree on channel order:

```python
# TwinLiteSegmenter.__call__  (line 126-129) — PyTorch
padded, ratio, (pl, pt) = letterbox(img_bgr, self.img_size)
t = torch.from_numpy(padded).to(self.device).float()      # still BGR

# TwinLiteTRTSegmenter.__call__ (line 177) — TensorRT
t = torch.from_numpy(canvas[:, :, ::-1].copy()).cuda()    # BGR -> RGB
```

One of them is wrong, and both are selected by the same
`segmentation_method: twinlitenet` config key — `create_segmenter` picks
between them purely on whether the weights path ends in `.engine`
(`segmentation.py:194-196`). So swapping in a TensorRT engine silently changes
the network's input, which is exactly the kind of difference that makes a
model "work on the laptop, not on the car."

Needs someone who knows how `nano.pth` was trained to say which order is
correct, then make both paths match. Worth an offline test that runs a fixed
image through both and asserts the masks agree.

Related: `DEPLOY.md` §6 measures TwinLiteNet at 73.7 ms and calls it the
bottleneck keeping the car at ~5 Hz (below the 8 Hz acceptance bar), and
`FINALIZE.md` Phase 2 is about TensorRT-exporting it. If the two paths
disagree on channel order, that export will change accuracy as well as speed,
and the change will look like a TensorRT problem.

### C3 — `real_nav2_params.yaml` marks the bridge's clearing endpoints as obstacles — FIXED (this branch)

`costmap_to_cloud.py` emits, on every bearing that hits nothing, a *clearing
endpoint* at exactly `raytrace_range_m` = **16.0 m** (`:43`, `:67-68`). It is
not an obstacle; it exists so Nav2 raytraces that bearing free. For that to
work, Nav2's `obstacle_max_range` must be **below** 16 m, or the endpoint gets
marked as a real obstacle.

`nav2_params_auto_drive.yaml:396` gets this right, and says so:

```yaml
obstacle_max_range: 15.5  # clearing endpoints are emitted at 16 m and are not marked
raytrace_max_range: 16.0
```

`real_nav2_params.yaml` does not:

| costmap | `obstacle_max_range` | 16 m endpoint |
|---|---|---|
| local (`:106`)  | 20.0 | **marked as an obstacle** |
| global (`:143`) | 30.0 | **marked as an obstacle** |

So under `deploy/real_nav2_launch.py`, every clear bearing plants a phantom
obstacle at 16 m, producing an arc of fake obstacles across the whole ±100°
fan — exactly the "wall the planner cannot get past" failure that
`costmap_to_cloud.py`'s docstring says the ray-casting design exists to avoid.

**Fixed here:** both costmaps in `real_nav2_params.yaml` now use
`obstacle_max_range: 15.5` / `raytrace_max_range: 16.0`, matching the auto_drive
config, with the reason in a comment. Not verified live (needs Nav2 running);
the arithmetic is hand-checkable and auto_drive is the precedent. Deriving both
from the bridge's own parameters would remove the drift risk entirely — worth
doing if these ever need to change.

### C4 — the default perception config silently disables road-keeping — FIXED (this branch)

`costmap_to_cloud.py` only forwards cells at or above `obstacle_threshold: 97`.
`real_nav2_params.yaml:89-93` warns about this in a comment:

> ROAD-KEEPING DEPENDENCY: off-road cells only reach Nav2 if perception marks
> them at/above costmap_to_cloud's OBST_THRESH (97). [...] If you lower
> offroad_cost below 97 (the perception_costmap.yaml default is 65), off-road
> stops entering the cloud and road-keeping silently disappears.

The warning describes the shipped state. `config/perception_costmap.yaml` — the
default config, the one `perception.launch.py` loads and the one the CARLA
smoke test in the README uses — **does not set `offroad_cost` at all**, so it
falls back to the node default of **65** (`costmap_node.py:196`,
`occupancy.py:40`). Only `perception_dinosaur.yaml:162` sets 97.

So anyone running the documented default pipeline gets obstacle avoidance but
no road-keeping, with nothing in the logs to say so. Two coupled constants in
two repos with no runtime check.

Fix options: set `offroad_cost: 97` in `perception_costmap.yaml` too; or have
the bridge log a warning at startup when the costmap it receives contains no
cells at or above its threshold; or publish the threshold as a topic/param the
bridge reads. The startup warning is the cheapest and catches every future
drift.

**Fixed here (both parts):** `perception_costmap.yaml` now sets
`offroad_cost: 97` with the coupling documented, and `costmap_to_cloud.py`
gained a runtime check that warns when a large sub-threshold plateau is seen.
Measured effect of the bug: 17 marked rays out of 401 at `offroad_cost: 65`
versus 386 at 97 — the road edges were simply absent from Nav2. Evidence and
both-direction verification in
`logs/results/2026-09-10_c3-c4-costmap-nav2-coupling.md`.

---

## Docs that do not match the code

Individually small; together they cost a new contributor a lot of time, and
`CLAUDE.md` opens with "Read this before trusting anything else in the repo."

### D1 — test counts are wrong in six places — FIXED (this branch)

Actual, on this branch:

```
perception_costmap  59 passed in 0.37s
driving_seg          7 passed in 0.09s
```

Claimed: **39** at `README.md:34`, `README.md:65`,
`ros2_ws/src/perception_costmap/README.md:66`, `CLAUDE.md:23`, `CLAUDE.md:54`,
`FINALIZE.md:25`; and **78** at
`ros2_ws/src/perception_costmap/README.md:133` — which contradicts the 39 in
the Tests section of that same file.

### D2 — five documented tools do not exist — FIXED by `0fedca3`

The package README's Tools table listed `tools/eval_ipm_calibration.py`,
`tools/benchmark_models.py`, `tools/measure_zed_sync.py`,
`tools/analyze_zed_depth.py` and `deploy/record_perception_bag.sh`, none of
which were committed. Same cause as B1: eleven files in all were untracked on
Alexander's machine. All are in `0fedca3`.

### D3 — `perception/` directory is referenced but absent — FIXED (this branch)

Cited at `README.md:37` ("Adam Castillo's original prototype scripts"),
`CLAUDE.md:37` ("Keep author credits intact"), `DESIGN.md:67`,
`segmentation.py:7`, and `obstacles.py:7`. There is no `perception/` directory
at the repo root on this branch.

Worth flagging to Alexander separately: `CLAUDE.md` asks that credit to Adam
Castillo be preserved, and the directory that carried it is gone. `.mailmap`
and `CONTRIBUTORS.md` still exist, so the attribution isn't lost, but the
files those docs point at are.

### D4 — `CLAUDE.md` points at a plan file that does not exist — FIXED (this branch)

`CLAUDE.md:40` cites `docs/plans/2026-07-01-perception-v2-sim-to-real.md`
("all 10 tasks complete"). The only plan in the tree is
`ros2_ws/docs/superpowers/plans/2026-07-16-perception-accuracy-prompt-and-plan.md`.

### D5 — costmap legend says off-road = 97, the node default is 65 — FIXED by C4

`ros2_ws/src/perception_costmap/README.md:11` documents the output as
"road = 0, caution = 1-96, off-road = 97, lethal = 100". But `offroad_cost`
defaults to **65** (`occupancy.py:40` `DEFAULT_OFFROAD_COST`, and
`costmap_node.py:196`). Only `config/perception_dinosaur.yaml:162` sets 97.

So the documented legend describes the car config, not the default the CARLA
smoke test in that same README will actually produce. Either say which config
the table describes, or make the defaults agree.

### D6 — `CLAUDE.md` says the work lives on `feature/alexander` — OPEN

`CLAUDE.md:19`. This branch is `copy`, which is 3 commits ahead of `main`.
`CONTRIBUTION_GUIDE.md` documents only `main` and `feature/<name>`, so `copy`
is undocumented. Worth confirming which branch is authoritative before anyone
opens a PR against the wrong base.

---

## Bloat / cleanup candidates

Nothing here is a bug. **None of it should be deleted without Alexander's
sign-off** — it is his repo and several items are deliberate.

### N1 — 22 MB duplicate model file — deliberate, but reconsider

`models/cone_det.pt` and `driving_seg/models/cone_det.pt` are byte-identical:

```
b4c1b4f3bfd4617ff0a15cfd539ad9f6  models/cone_det.pt
b4c1b4f3bfd4617ff0a15cfd539ad9f6  driving_seg/models/cone_det.pt
```

`models/README.md:27` says this is intentional, so each package stays
self-contained. That is a real reason. The cost is 22 MB duplicated in every
clone forever (44 MB of the 67 MB working tree; `.git` is another 42 MB), and
a silent drift risk if one copy is ever retrained and the other is not.

Options, in increasing order of disruption: leave it and add a checksum test
that fails when the two diverge; symlink one to the other; move both behind
git-lfs. The first is cheap and removes the drift risk without touching
anyone's workflow.

### N2 — six stub ROS packages — leave alone, but they are not free

`collision_guard`, `controller`, `route_planner`, `sdc_bringup`, `sdc_common`,
`world_setup`. `CLAUDE.md`, `ros2_ws/README.md`, and the root README all warn
against building on them. Each ships identical boilerplate
`test_copyright.py` / `test_flake8.py` / `test_pep257.py`, so a bare
`colcon build` and `colcon test` still walk them.

They are small on disk. The real cost is that every newcomer has to be told
three times which packages are fake. Not our call to delete — the point of
listing it here is that if we ever *do* clean up, this is the highest-value
target, and `colcon build --packages-select perception_costmap` (already the
documented command) is the zero-risk workaround in the meantime.

### N3 — `.pytest_cache/` and `__pycache__/` in the working tree — no action

Present locally, correctly gitignored, not tracked. Noted only so nobody
"fixes" it twice.

---

## Built here, beyond the bug fixes

### `/perception/reset` — between-runs state reset (IGVC phase 1)

`costmap_node.py` had no reset path of any kind: the `TemporalObstacleFilter`
confidence arrays accumulate across ticks by design and live for the life of
the process, so a second run in the same process started with the first run's
obstacles still confirmed. The only clear was restarting the whole stack, which
worked by accident and is not a between-runs procedure.

Added:
- `TemporalObstacleFilter.reset()` (`temporal.py`) — ROS-free, returns the
  count of cells that were reporting lethal so a caller can say what it threw
  away instead of claiming success blindly. 3 tests.
- `util.clear_sample_buffer()` — empties a buffer's `.samples` deque
  (`sample_buffer`'s classes have no `clear()` of their own). 2 tests.
- `/perception/reset` (`std_srvs/Trigger`) in `costmap_node.py` — 58 lines,
  self-contained.
- `deploy/fresh_run.sh` — calls the reset plus both Nav2 costmap clears,
  reports per-step status, and exits non-zero telling you not to start a scored
  run. Verified in all three branches: service missing, service OK, and service
  answering `success=False` (which `ros2 service call` reports with exit code
  0 — the payload has to be checked, not the exit status).

**Caveat:** the handler is covered by 5 static AST tests, the important one
asserting that every attribute the handler assigns already exists in
`__init__` — a typo there would silently create a new attribute and leave the
real one stale, the actual failure mode of attribute-based reset code. That
test was verified to fail on an injected typo. A live call on the car is in
`DEPLOY.md`'s test checklist once that lands.

## Verified working (so nobody re-checks)

- `git clone` + `git checkout copy` — clean, no LFS, no submodules.
- `perception_costmap` offline suite: **59 passed**, 0.37 s, numpy 2.4.6 +
  pytest 7.4.4, no ROS sourced. Pure-Python/numpy/opencv as `CLAUDE.md`
  promises.
- `driving_seg` offline suite: **7 passed**, 0.09 s, with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Rebuilt on Alexander's `0fedca3` (2026-09-14): `perception_costmap` suite
  **90 passed** (his 78 + 12 from this branch: C1, reset, temporal, util), the
  node imports under ROS 2 Jazzy, and the bridge publishes
  `/perception/costmap_cloud` at 9.97 Hz with his `costmap_cloud.py`.
- Every relative import in `driving_seg/driving_seg/` resolves.
- **The campus map is fine — it is not the cause of the click rejection.**
  `graph_validation.py` on `cpp_campus_graph.geojson`: 9112 nodes, 17492
  edges, one weak component (nothing stranded), 9082 nodes in the
  bidirectionally routable core, 20/20 route probes succeeded over 25.1 km.
  **PASS.** One warning worth knowing: 30 nodes sit outside the routable core
  (one-way fringe), so a destination clicked right on one of those could still
  fail to route — unrelated to B2.

Not yet attempted here: `colcon build` (this box is ROS 2 **Jazzy**, the
project targets **Humble**), CARLA smoke test, anything on the Jetson.

---

## Camera launch and performance (2026-09-14)

Found while building `launch/perception_stack.launch.py`. Profile evidence:
`logs/results/2026-09-14_perception-profile-and-zed-audit.md`.

### C8 — left and right ZED serials are swapped between launch paths — OPEN

| camera | `deploy/full_stack_restart.sh`, `deploy/clean_camera_restart.sh` (boot + recovery) | `how_dinosaur_drives/lidar/sensors.launch.py` (from IGVC_ROS2) |
|---|---|---|
| front | 42569280 | 42569280 |
| left  | **49910017** | **43779087** |
| right | **43779087** | **49910017** |

Start the cameras through the other path and `/zed_left` publishes the right
camera's images: side obstacles land on the wrong side of the costmap, and
every yaw calibration is 180° off. `perception_stack.launch.py` follows the
boot scripts, since the 2026-07-09 calibration was done on that path.
`sensors.launch.py` even carries `TODO: confirm right-camera serial`. **Needs
someone at the car** to cover one camera and see which topic goes dark.

### C9 — the car subscribes to a confidence map the ZED never advertises — NOT A BUG on the car (checked 2026-09-15)

Suspected from the GitHub copy of avros_bringup's camera configs:
zed-ros2-wrapper 5.2 only creates `confidence/confidence_map` when
`depth.publish_depth_confidence: true` (default false), and those configs never
set it, while `perception_dinosaur.yaml` subscribes to it.

**On the car it's fine.** The installed configs
(`~/IGVC/install/avros_bringup/share/avros_bringup/config/zed_*.yaml`) set
`publish_depth_confidence: true`, and all three `confidence_map` topics are
advertised (`logs/results/2026-09-15_car-test-dinosaur.md`). The GitHub copy of
IGVC_ROS2 (last pushed 2026-06-01) is behind the car. The lean profiles keep
the setting on explicitly, so they don't depend on the wrapper default.

Lesson: audit the files installed on the car, not the GitHub copy.

### P1 — ZED drivers compute things nothing reads — FIXED in `config/zed_perception_*.yaml`

The avros_bringup camera configs were tuned for an older consumer, the
kiwicampus semantic layer in Nav2, which read ZED point clouds. Nothing on the
current stack reads a ZED point cloud (`deploy/depth_obstacle_node.py` does,
but the boot script doesn't start it), yet every camera still runs:

- **positional tracking**, kept on "because the cloud needs it". In wrapper
  5.2 it also forces depth to be computed on every grab, with or without
  subscribers, and the Xsens/EKF already own localization;
- depth stabilization (temporal smoothing, requires positional tracking);
- depth at the 15 fps grab rate, when the front publishes at 8;
- side cameras publishing 15 Hz, when perception runs YOLO on each side
  camera every 0.4 s. The node still converts every frame through cv_bridge.

The lean profiles publish RGB, depth and confidence only; turn off point
clouds, positional tracking, IMU and odometry; and cap processing at 8 fps.
**Confirmed on the car (2026-09-15):** the installed configs set
`pos_tracking_enabled: true`, and each camera advertises 27 topics including
`point_cloud` and `odom`/`pose`. Whether turning them off frees measurable CPU/GPU
is what the lean-launch car test (`DEPLOY.md` §7 step 3) measures.

**Measured on the car 2026-09-16** (left + right; the front camera was held
open by another program at the time -- see the log -- so it was left out).
Same costmap node and cameras in both runs, only the camera YAML differs:

| | old configs | lean profile |
|---|---|---|
| both camera drivers | 50% of a core | **31%** (-38%) |
| costmap_node | 122% | **108%** |
| CPU, 12-core mean | 24.8% | **22.4%** |
| GPU mean | 59.2% | 56.9% |
| RAM | 6096 MB | **5918 MB** |
| ZED topics per camera | 27 | **23** |
| /perception/costmap | 9.64 Hz | **9.99 Hz** |

About a third of a core freed on two cameras, mostly
from publishing at 8 Hz instead of 15 and dropping the point cloud and
positional tracking. GPU is nearly unchanged: depth still runs per grab.
With three cameras the old profile costs **80% of a core** in camera drivers
(front 26% at 8 Hz, left 28%, right 26% at 15 Hz) -- a camera's cost is mostly
the point cloud and positional tracking, not its rate -- so a lean three-camera
run should land near 45-48%. That run is still to do; the boot stack could not
be stopped while teammates were using its topics.

Details: `logs/results/2026-09-16_lean-vs-old-camera-profile.md`; the first,
incomplete attempt: `logs/results/2026-09-15_lean-launch-car-attempt.md`.

All 94 keys were checked against wrapper v5.2.2's parameter tree. **Not yet
run on the car.** `depth_stabilization: 0` is the one real tradeoff: if depth
gets too noisy, set 1 and turn positional tracking back on.

### P2 — the costmap node's main thread is its bottleneck — ON HOLD until car testing

Profiled with 3 cameras at the car's rates (front 8 Hz, sides 15 Hz,
960x600), the car's config minus the TensorRT models, laptop CPU, on
`copy` @ `0fedca3` plus this repo's C5/C6 fixes: the node needs **149% of a
core** to hold 10 Hz, and the main-thread stages below cost **~28 ms per tick**
(of 100 ms). The Orin's CPU cores are several times slower. Alexander's own
py-spy run (`d5dac06`) found the same thing: main thread ~84% busy, CPU bound
rather than GPU bound.

| work, per tick | before `d5dac06` | now | why it costs |
|---|---|---|---|
| HSV road segmentation, 3 cameras | 21.5 ms | **18.1 ms** | runs on every camera **every tick**, even when the frame hasn't changed |
| white-line mask, 3 cameras | 5.2 | **5.1** | same: every camera, every tick |
| `build_cost_array` | 4.5 | **3.0** | `cv2.inpaint` for blind-spot infill |
| grid reprojection | 3.8 | **1.8** | `d5dac06` builds the sampling maps once per observation |
| `np.isin` in segmentation | 3.2 | **0** | `d5dac06` replaced it with a lookup table |

Already done by Alexander in `d5dac06`: the `np.isin` lookup table and shared
reprojection maps. Biggest remaining candidate: skip segmentation and the
white-line mask when a camera's stamp hasn't changed (about 20% of front-camera
ticks at 8 Hz; more once the side cameras drop to 8 Hz with the lean profiles).

**On hold (team decision, 2026-09-14):** no costmap performance changes until
PRs #1-#4 have been tested on the car (`DEPLOY.md` §7). The laptop profile
shows where the time goes; the car's numbers decide which fixes are worth it.

### P3 — `DEPLOY.md` still blames TwinLiteNet, which the car no longer runs — FIXED (docs)

`DEPLOY.md` §6 and `full_stack_restart.sh` ("TRT engine + TwinLiteNet") name
TwinLiteNet's 73.7 ms as the bottleneck, but `perception_dinosaur.yaml` now sets
`segmentation_method: hsv`. The real bottleneck is P2. `DEPLOY.md` §6 now
marks those numbers as historical, and §7 re-measures on the car.
`full_stack_restart.sh`'s comment is left as-is: it's the as-run boot script
and isn't being edited until the car test.

### P4 — the boot service starts viewers nobody may be watching — OPEN

`percept-stack.service` → `full_stack_restart.sh` starts, on every boot:
`viz_node` (subscribes to all 3 RGB streams and renders a BEV composite),
`costmap_rgb_node` (a 40,000-point colored cloud), `web_video_server`, an
HTTP server, and **RViz with software OpenGL (llvmpipe)** showing the costmap
plus 3 camera panels, whenever a NoMachine display exists.
`auto_drive.launch.py` opens a second software-rendered RViz. RViz on
llvmpipe renders on the CPU the controller and perception share, and its
camera panels keep all three RGB streams flowing. `perception_stack.launch.py`
starts none of these; run them only while someone is watching. The boot
service itself is unchanged until the lean launch has run on the car.
