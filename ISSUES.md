# ISSUES — bugs and open problems on this branch

Working log of what is broken, found while setting up a **fresh clone** of
`copy` on a laptop (Ubuntu, ROS 2 Jazzy, no GPU sensors attached) on
2026-09-10. `FINALIZE.md`'s standing rules say to "verify packaging with a
FRESH CLONE (three bugs caught that way)" — B1 below is a fourth.

Status key: **OPEN** / **IN PROGRESS** / **FIXED** (commit) / **WONTFIX** (why).

Evidence for each entry goes in `logs/` — see `logs/README.md`.

---

## Blockers

### B1 — `costmap_node.py` imports four modules that do not exist — OPEN

The main ROS node cannot be imported, so it cannot start.

`ros2_ws/src/perception_costmap/perception_costmap/costmap_node.py:32-35`

```python
from .detection_schedule import DetectionScheduler
from .inference_worker import LatestTaskWorker
from .sample_buffer import PoseBuffer, TimestampedBuffer
from .health import camera_health
```

None of `detection_schedule.py`, `inference_worker.py`, `sample_buffer.py`,
`health.py` exist in the package. They are not in any commit on any branch:

```
$ git log --all --oneline -- '*perception_costmap/sample_buffer.py'
(no output — same for the other three)
```

Reproduce:

```
$ source /opt/ros/jazzy/setup.bash
$ cd ros2_ws/src/perception_costmap && PYTHONPATH=. python3 -c "import perception_costmap.costmap_node"
ModuleNotFoundError: No module named 'perception_costmap.detection_schedule'
```

Consequences:
- `setup.py`'s console script `costmap_node = perception_costmap.costmap_node:main`
  is dead, so `ros2 launch perception_costmap perception.launch.py` fails.
- Everything in `REPRODUCE.md` step 7 and `DEPLOY.md` §5 is unreachable.
- **The 59 offline tests still pass**, because not one of them imports
  `costmap_node`. The suite covers only the ROS-free helper modules. Green
  tests are not evidence the node runs.

Symbols actually used (so we know the shape of what is missing):

| symbol | used at | needs to provide |
|---|---|---|
| `TimestampedBuffer(maxlen=)` | `costmap_node.py:55,56` | `.add(stamp, value)`, `.nearest(stamp, tol) -> (stamp, value) \| None`, `.samples` |
| `DetectionScheduler(primary=, secondary_stride=)` | `:335,338` | `.select(names) -> list[str]` |
| `PoseBuffer(maxlen=)` | `:372` | `.add(stamp, pose)`, `.interpolate(stamp, tol) -> (x,y,yaw) \| None`, `.samples` |
| `LatestTaskWorker(fn)` | `:398` | `.submit(task)`, `.take_latest()`, `.take_error()`, `.close(timeout=)`, `.submitted/.replaced/.completed` |
| `camera_health(...)` | `:802` | `(image_age, depth_age, confidence_age, stale, conf_expected) -> (level:int, message:str)` |

Ask Alexander whether these exist uncommitted on his machine before writing
replacements — the README "Dinosaur accuracy upgrade (2026-09-03)" section
describes this exact worker/scheduler design as done and running on the car,
so the files probably exist and simply were never `git add`ed.

**Do not start here without asking.** Reimplementing from scratch risks
throwing away working, on-car-validated code.

---

## Code bugs

### C1 — `GridSpec.world_to_cell` truncates toward zero, admitting out-of-grid points — OPEN

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

Fix is `int(np.floor(...))`.

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

---

## Docs that do not match the code

Individually small; together they cost a new contributor a lot of time, and
`CLAUDE.md` opens with "Read this before trusting anything else in the repo."

### D1 — test counts are wrong in six places — OPEN

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

### D2 — five documented tools do not exist — OPEN

The Tools table in `ros2_ws/src/perception_costmap/README.md` lists these, and
the "Still needs field data" section tells you to run two of them:

- `tools/eval_ipm_calibration.py`
- `tools/benchmark_models.py`
- `tools/measure_zed_sync.py`
- `tools/analyze_zed_depth.py`
- `deploy/record_perception_bag.sh`

`tools/` actually contains: `bench_perception.py`, `carla_feed.py`,
`eval_road_iou.py`, `export_trt.py`, `ipm_overlay.py`,
`shadow_robustness_test.py`, `viz_node.py`.

### D3 — `perception/` directory is referenced but absent — OPEN

Cited at `README.md:37` ("Adam Castillo's original prototype scripts"),
`CLAUDE.md:37` ("Keep author credits intact"), `DESIGN.md:67`,
`segmentation.py:7`, and `obstacles.py:7`. There is no `perception/` directory
at the repo root on this branch.

Worth flagging to Alexander separately: `CLAUDE.md` asks that credit to Adam
Castillo be preserved, and the directory that carried it is gone. `.mailmap`
and `CONTRIBUTORS.md` still exist, so the attribution isn't lost, but the
files those docs point at are.

### D4 — `CLAUDE.md` points at a plan file that does not exist — OPEN

`CLAUDE.md:40` cites `docs/plans/2026-07-01-perception-v2-sim-to-real.md`
("all 10 tasks complete"). The only plan in the tree is
`ros2_ws/docs/superpowers/plans/2026-07-16-perception-accuracy-prompt-and-plan.md`.

### D5 — costmap legend says off-road = 97, the node default is 65 — OPEN

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

## Verified working (so nobody re-checks)

- `git clone` + `git checkout copy` — clean, no LFS, no submodules.
- `perception_costmap` offline suite: **59 passed**, 0.37 s, numpy 2.4.6 +
  pytest 7.4.4, no ROS sourced. Pure-Python/numpy/opencv as `CLAUDE.md`
  promises.
- `driving_seg` offline suite: **7 passed**, 0.09 s, with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Every relative import in `driving_seg/driving_seg/` resolves.

Not yet attempted here: `colcon build` (this box is ROS 2 **Jazzy**, the
project targets **Humble**), CARLA smoke test, anything on the Jetson.
