# Perception v2 — Accurate CV → Costmap, Sim-to-Real Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `perception_costmap` package produce an *accurate* Nav2 costmap from real CV models (TwinLiteNet+ for road, YOLOv8 for objects, lidar for metric truth), across the real car's 3-camera layout, with the exact same code path running in CARLA and on the Jetson.

**Architecture:** Keep the current split — ROS-free perception modules (`segmentation`, `obstacles`, `bev`, `occupancy`) + one thin ROS node. Improvements land as: (1) transport correctness fixes (QoS, staleness, vectorization), (2) a temporal filter so single-frame noise never reaches Nav2, (3) learned models behind the existing interfaces, loaded once, (4) multi-camera BEV fusion with yaw-aware homographies, (5) calibration/eval tooling so "accurate" is measured, not claimed, (6) a Jetson deploy path (TensorRT engines, benchmark, checklist).

**Tech Stack:** ROS2 (Humble target, Foxy/Jazzy tolerant), Python, OpenCV, numpy; optional lazy deps: ultralytics (YOLOv8), torch (TwinLiteNet+), CARLA 0.9.16 Python API (sim only, 5090 box only).

## Global Constraints

- Python 3.8-compatible syntax everywhere (Jetson JetPack 5 = Ubuntu 20.04). No `match`, no `X | Y` type unions, no `list[int]` hints.
- `torch`, `ultralytics`, `carla`, ROS message types stay **lazy imports**. `pytest` in `test/` must pass on a machine with only numpy + opencv.
- Core modules stay ROS-free; only `costmap_node.py` (and tools that explicitly say so) import rclpy.
- All grid math follows REP-103 (+x forward, +y left) and `OccupancyGrid` row-major `index = row*width + col`. Costs: -1 unknown / 0 free / 100 lethal.
- Every task ends with `PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q` green and a commit on `feature/alexander`. Commit style: `perception_costmap: <what>`. No AI co-author lines.
- Working dir for all commands: `~/carla-nav2-backup/ros2_ws/src/perception_costmap` unless stated.

## Why these changes (the accuracy argument, in one place)

1. **QoS mismatch silently drops every frame.** CARLA's ROS2 output and real camera/lidar drivers publish BEST_EFFORT; our subscriptions are default RELIABLE. On some DDS pairings the node receives *nothing* and just never publishes. This is the #1 sim-to-real killer and it's a 3-line fix.
2. **A full YOLO box through IPM smears obstacles across 10+ m.** IPM assumes pixels lie on the ground plane. A car's roof pixels are 1.5 m up, so warping the whole box projects "obstacle" far behind the car. Only the *bottom strip* of a box touches the ground — rasterize that.
3. **`YOLO("yolov8n.pt")` inside the detect function reloads the network every frame.** That alone makes YOLO unusable on the Jetson. Load once, reuse; accept `.engine` weights so the same class runs TensorRT on the Orin.
4. **Single-frame masks flicker.** One noisy frame paints a lethal cell in front of the planner and it swerves. A tiny confidence accumulator (2 hits to mark, ~3 misses to clear) kills flicker with ~200 ms latency at 10 Hz.
5. **One forward camera ≠ the real car.** The car has a 3-camera layout. Side cameras need yaw in the homography and per-camera calibration; the BEV grid is where they fuse for free.
6. **"Accurate" needs a number.** CARLA's semantic camera is free ground truth. Log paired frames, compute road-mask IoU (image space and BEV space) for HSV vs TwinLiteNet+. That's the evidence for choosing the model per task.
7. **Per-point Python loop over lidar** (`points_to_grid_mask`) is O(N) interpreted — fine at 500 points, not at 50k. Vectorize.

## File Structure

```
ros2_ws/src/perception_costmap/
  perception_costmap/
    util.py            NEW  stamp helpers (pure)
    temporal.py        NEW  TemporalObstacleFilter (pure numpy)
    segmentation.py    MOD  Segmenter classes + factory + letterbox
    obstacles.py       MOD  footprint rasterizer, YoloObstacleDetector, vectorized binning
    bev.py             MOD  homography_from_extrinsics (adds yaw + xy offset)
    occupancy.py       (unchanged)
    costmap_node.py    MOD  QoS, staleness, CameraSource multi-cam, model warm-load, filter
  config/
    perception_costmap.yaml   MOD  cameras list + per-camera blocks + new params
    nav2_costmap_params.yaml  (unchanged)
  tools/               NEW  (plain scripts, not console entries)
    ipm_overlay.py          draw the BEV grid back onto a camera frame (calibration check)
    carla_feed.py           CARLA 0.9.16 → ROS2 topics + paired-frame logger (5090 box)
    eval_road_iou.py        HSV vs TwinLiteNet+ vs semantic ground truth, image + BEV IoU
    export_trt.py           YOLO .pt → TensorRT .engine (run ON the Jetson)
    bench_perception.py     per-stage ms + FPS on any machine (laptop / 5090 / Jetson)
  test/
    test_util.py test_temporal.py test_footprint.py test_segmenter.py
    test_carla_convert.py  NEW; test_geometry.py test_detect.py extended
  DEPLOY.md            NEW  Jetson Orin Nano bring-up checklist
  README.md DESIGN.md  MOD  final status/dataflow update
```

---

### Task 1: Sensor QoS + stale-frame guard

The node must subscribe with sensor-data QoS (BEST_EFFORT, volatile) or it can receive zero frames from CARLA/real drivers. And it must stop building costmaps from a frozen frame when a camera dies — stale data is worse than no data.

**Files:**
- Create: `perception_costmap/util.py`
- Modify: `perception_costmap/costmap_node.py`
- Test: `test/test_util.py`

**Interfaces:**
- Produces: `util.stamp_to_sec(stamp) -> float`, `util.is_fresh(stamp_sec: float, now_sec: float, max_age: float) -> bool`. Task 6's `CameraSource` reuses both.

- [ ] **Step 1: Write the failing test**

```python
# test/test_util.py
from types import SimpleNamespace
from perception_costmap.util import stamp_to_sec, is_fresh


def test_stamp_to_sec():
    stamp = SimpleNamespace(sec=12, nanosec=500_000_000)
    assert abs(stamp_to_sec(stamp) - 12.5) < 1e-9


def test_is_fresh_within_budget():
    assert is_fresh(stamp_sec=10.0, now_sec=10.3, max_age=0.5)


def test_is_stale_past_budget():
    assert not is_fresh(stamp_sec=10.0, now_sec=10.6, max_age=0.5)
```

- [ ] **Step 2: Run it — expect FAIL** (`ModuleNotFoundError: perception_costmap.util`)

Run: `PYTHONPATH=.:$PYTHONPATH python3 -m pytest test/test_util.py -q`

- [ ] **Step 3: Implement `util.py`**

```python
"""
util.py — small pure helpers shared by the node and tools.
"""


def stamp_to_sec(stamp):
    """builtin_interfaces/Time (or anything with .sec/.nanosec) -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def is_fresh(stamp_sec, now_sec, max_age):
    """True if a sample stamped at stamp_sec is younger than max_age at now_sec."""
    return (now_sec - stamp_sec) <= max_age
```

- [ ] **Step 4: Run test — expect PASS**, then wire the node:

In `costmap_node.py`:
- add imports: `from rclpy.qos import qos_profile_sensor_data` and `from .util import stamp_to_sec, is_fresh`
- declare two new parameters in the list: `("image_stale_sec", 0.5), ("lidar_stale_sec", 0.5)`; stash as `self.img_stale, self.lidar_stale`
- every `create_subscription(...)` gets `qos_profile_sensor_data` instead of `1`
- `_on_image` also stores `self._img_stamp = stamp_to_sec(msg.header.stamp)`; `_on_lidar` stores `self._pts_stamp` the same way
- in `_tick`, compute `now = stamp_to_sec(self.get_clock().now().to_msg())`; the camera branch requires `self._latest_img is not None and is_fresh(self._img_stamp, now, self.img_stale)`; the lidar branch requires the analogous check. (With `use_sim_time` set, node clock and CARLA stamps share the same timeline — mention that in the param comment.)
- add both params to `config/perception_costmap.yaml` under a `# --- freshness ---` comment.

- [ ] **Step 5: Full test suite green, then commit**

Run: `PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q` → all pass (11 old + 3 new)

```bash
git add perception_costmap/util.py perception_costmap/costmap_node.py config/perception_costmap.yaml test/test_util.py
git commit -m "perception_costmap: sensor-data QoS + stale-frame guard"
```

---

### Task 2: Vectorize lidar → grid binning

**Files:**
- Modify: `perception_costmap/obstacles.py` (`points_to_grid_mask`)
- Test: `test/test_geometry.py` (append)

**Interfaces:**
- Produces: same signature `points_to_grid_mask(points_xyz, grid) -> bool ndarray (h, w)` — callers unchanged.

- [ ] **Step 1: Append the failing-fast test (it passes today — it pins behavior; the perf change must keep it green)** plus an empty-input case:

```python
# append to test/test_geometry.py
def test_points_to_grid_mask_vectorized_matches_cells():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    pts = np.array([[0.5, 0.5, 1.0],    # cell (col 0, row 0)
                    [1.5, 0.5, 1.0],    # cell (col 1, row 0)
                    [9.0, 9.0, 1.0]])   # out of grid, dropped
    m = points_to_grid_mask(pts, g)
    assert m.shape == (2, 2)
    assert m[0, 0] and m[0, 1]
    assert m.sum() == 2


def test_points_to_grid_mask_empty():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    m = points_to_grid_mask(np.zeros((0, 3)), g)
    assert m.shape == (2, 2) and not m.any()
```

(Import `points_to_grid_mask` at the top of the test file if it isn't already.)

- [ ] **Step 2: Run — both should pass against the loop version.** Now replace the body:

```python
def points_to_grid_mask(points_xyz: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Bin metric obstacle points into a grid-space boolean obstacle mask.
    Vectorized: floor to cell indices, keep in-bounds, scatter."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    if points_xyz.size == 0:
        return mask
    cols = np.floor((points_xyz[:, 0] - grid.x_min) / grid.resolution).astype(np.int64)
    rows = np.floor((points_xyz[:, 1] - grid.y_min) / grid.resolution).astype(np.int64)
    ok = (cols >= 0) & (cols < grid.width) & (rows >= 0) & (rows < grid.height)
    mask[rows[ok], cols[ok]] = True
    return mask
```

- [ ] **Step 3: Run full suite — PASS.** Commit:

```bash
git add perception_costmap/obstacles.py test/test_geometry.py
git commit -m "perception_costmap: vectorize lidar point binning"
```

---

### Task 3: Temporal obstacle filter (no more single-frame flicker)

A per-cell confidence in [0,1]: obstacle detections add `hit`, observed-empty cells subtract `miss`. A cell is reported lethal only at `conf >= threshold`. Defaults: 2 consecutive hits to mark, ~3 misses to clear (≈200–300 ms at 10 Hz).

**Files:**
- Create: `perception_costmap/temporal.py`
- Modify: `perception_costmap/costmap_node.py`
- Test: `test/test_temporal.py`

**Interfaces:**
- Produces: `TemporalObstacleFilter(shape, hit=0.4, miss=0.2, threshold=0.5)` with `update(obstacle_mask, observed_mask) -> bool ndarray`. Node calls `update()` once per `_tick` on the *fused* obstacle grid.

- [ ] **Step 1: Failing test**

```python
# test/test_temporal.py
import numpy as np
from perception_costmap.temporal import TemporalObstacleFilter


def _masks(shape, obstacle_cells):
    obs = np.zeros(shape, bool)
    for r, c in obstacle_cells:
        obs[r, c] = True
    seen = np.ones(shape, bool)
    return obs, seen


def test_needs_two_hits_to_confirm():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    assert not f.update(obs, seen)[1, 1]      # 1st hit: not yet
    assert f.update(obs, seen)[1, 1]          # 2nd hit: confirmed


def test_clears_after_misses():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    f.update(obs, seen); f.update(obs, seen); f.update(obs, seen)  # conf -> 1.0
    empty = np.zeros((3, 3), bool)
    for _ in range(3):
        out = f.update(empty, seen)
    assert not out[1, 1]                      # 1.0 - 3*0.2 = 0.4 < 0.5


def test_unobserved_cells_hold_confidence():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    f.update(obs, seen); f.update(obs, seen)
    unseen = np.zeros((3, 3), bool)
    out = f.update(np.zeros((3, 3), bool), unseen)   # camera looked away
    assert out[1, 1]                           # still lethal — no evidence it left
```

- [ ] **Step 2: Run — FAIL (no module).**

- [ ] **Step 3: Implement `temporal.py`**

```python
"""
temporal.py — per-cell obstacle confidence over time.

A costmap built from single frames flickers: one noisy frame paints a lethal
cell and the planner reacts. This accumulator requires evidence to mark AND
evidence to clear:

  hit:   cell detected as obstacle this frame        conf += hit
  miss:  cell observed (in FOV / lidar) and empty    conf -= miss
  else:  not observed                                conf unchanged

Report lethal where conf >= threshold. Pure numpy, ROS-free.
"""

import numpy as np


class TemporalObstacleFilter:
    def __init__(self, shape, hit=0.4, miss=0.2, threshold=0.5):
        self.hit = float(hit)
        self.miss = float(miss)
        self.threshold = float(threshold)
        self.conf = np.zeros(shape, dtype=np.float32)

    def update(self, obstacle_mask, observed_mask):
        obstacle_mask = obstacle_mask.astype(bool)
        # decay anywhere we looked and saw nothing — including cells that only
        # ever had lidar evidence (conf > 0) so stale marks can't live forever
        decay = (observed_mask.astype(bool) & ~obstacle_mask)
        self.conf[obstacle_mask] += self.hit
        self.conf[decay] -= self.miss
        np.clip(self.conf, 0.0, 1.0, out=self.conf)
        return self.conf >= self.threshold
```

- [ ] **Step 4: Tests pass. Wire into the node:**

In `costmap_node.py.__init__`: params `("temporal_hit", 0.4), ("temporal_miss", 0.2), ("temporal_threshold", 0.5), ("temporal_enabled", True)`; create `self.obs_filter = TemporalObstacleFilter((self.grid.height, self.grid.width), hit=..., miss=..., threshold=...)` (import at top — pure numpy, safe).

In `_tick`, after fusing `obst_grid` and computing `known`. A 360° lidar observes the whole 20×20 m grid, so when a fresh lidar frame contributed this tick the observed mask is everything; otherwise it's the camera FOV:

```python
lidar_active = (self.use_lidar and self._latest_points is not None
                and is_fresh(self._pts_stamp, now, self.lidar_stale))
observed = (known if known is not None
            else np.zeros((self.grid.height, self.grid.width), bool))
if lidar_active:
    observed = np.ones_like(observed)
if self.temporal_enabled:
    obst_grid = self.obs_filter.update(obst_grid, observed)
```

Add the four params to `config/perception_costmap.yaml`.

- [ ] **Step 5: Full suite green → commit**

```bash
git add perception_costmap/temporal.py perception_costmap/costmap_node.py config/perception_costmap.yaml test/test_temporal.py
git commit -m "perception_costmap: temporal obstacle confidence filter"
```

---

### Task 4: YOLO done right — load once, footprint strip, wired into the node

**Files:**
- Modify: `perception_costmap/obstacles.py` (add `boxes_to_footprint_mask`, `YoloObstacleDetector`; reimplement `detect_obstacles_yolo` on top; keep signature)
- Modify: `perception_costmap/costmap_node.py`, `config/perception_costmap.yaml`
- Test: `test/test_footprint.py`

**Interfaces:**
- Produces: `boxes_to_footprint_mask(boxes_xyxy, image_shape, footprint_frac=0.25) -> bool ndarray` (pure, offline-testable); `YoloObstacleDetector(weights="yolov8n.pt", classes=(...), conf=0.35, footprint_frac=0.25, device=None)` with `.detect(img_bgr) -> bool ndarray`. Accepts `.pt` or `.engine` weights (ultralytics loads TensorRT engines transparently — this is the Jetson path, see Task 9).

- [ ] **Step 1: Failing test (pure rasterizer — no ultralytics needed)**

```python
# test/test_footprint.py
import numpy as np
from perception_costmap.obstacles import boxes_to_footprint_mask


def test_only_bottom_strip_is_marked():
    # box 40 px tall at rows 10..50; frac 0.25 -> bottom 10 rows (40..50)
    m = boxes_to_footprint_mask([(20, 10, 60, 50)], (100, 100, 3), footprint_frac=0.25)
    assert m[45, 30] and m[49, 30]        # inside the strip
    assert not m[20, 30]                  # roof of the box: NOT an obstacle cell
    assert not m[45, 10]                  # left of the box


def test_clips_to_image_and_min_one_row():
    m = boxes_to_footprint_mask([(-5, -5, 8, 3)], (10, 10), footprint_frac=0.1)
    assert m.shape == (10, 10)
    assert m[2, 4]                        # at least one row survives, clipped


def test_empty_boxes():
    assert not boxes_to_footprint_mask([], (5, 5)).any()
```

- [ ] **Step 2: Run — FAIL (ImportError).**

- [ ] **Step 3: Implement in `obstacles.py`**

```python
def boxes_to_footprint_mask(boxes_xyxy, image_shape, footprint_frac=0.25):
    """
    Rasterise detector boxes as *ground-contact strips*, not full boxes.

    IPM assumes every pixel lies on the ground plane. A vehicle's upper pixels
    are 1-2 m above it, so warping a full box smears "obstacle" many metres
    down-range. Only the bottom footprint_frac of each box (where the object
    meets the road) is geometrically valid to project.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        strip = max(1, int(round((y2 - y1) * footprint_frac)))
        r0, r1 = max(0, y2 - strip), min(h, y2)
        c0, c1 = max(0, x1), min(w, x2)
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask


class YoloObstacleDetector:
    """YOLOv8 wrapper that loads the network ONCE (the old per-call load was
    ~100x slower than inference itself). Pass a .engine file to run TensorRT
    on the Jetson — ultralytics handles both formats."""

    DEFAULT_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")

    def __init__(self, weights="yolov8n.pt", classes=DEFAULT_CLASSES,
                 conf=0.35, footprint_frac=0.25, device=None):
        from ultralytics import YOLO          # lazy: optional dependency
        self.model = YOLO(weights)
        self.classes = set(classes)
        self.conf = conf
        self.footprint_frac = footprint_frac
        self.device = device

    def detect(self, img_bgr):
        res = self.model(img_bgr, verbose=False, conf=self.conf,
                         device=self.device)[0]
        boxes = [b.xyxy[0].tolist() for b in res.boxes
                 if res.names[int(b.cls[0])] in self.classes]
        return boxes_to_footprint_mask(boxes, img_bgr.shape, self.footprint_frac)


def detect_obstacles_yolo(img_bgr, classes=YoloObstacleDetector.DEFAULT_CLASSES):
    """One-shot convenience kept for scripts. For anything per-frame use
    YoloObstacleDetector so the model loads once."""
    return YoloObstacleDetector(classes=classes).detect(img_bgr)
```

- [ ] **Step 4: Node wiring.** In `costmap_node.py`:
- new param `("obstacle_method", "classical")` — `classical | yolo | both`; plus `("yolo_weights", "yolov8n.pt"), ("yolo_conf", 0.35), ("yolo_footprint_frac", 0.25)`
- in `__init__`, if method includes yolo, build it eagerly (models must warm-load at startup, never mid-drive) with a graceful fallback:

```python
self.yolo = None
if g["obstacle_method"] in ("yolo", "both"):
    try:
        self.yolo = obstacles.YoloObstacleDetector(
            weights=g["yolo_weights"], conf=g["yolo_conf"],
            footprint_frac=g["yolo_footprint_frac"])
        self.get_logger().info("YOLO obstacle detector loaded: %s" % g["yolo_weights"])
    except ImportError:
        self.get_logger().warn(
            "obstacle_method=%s but ultralytics not installed; "
            "falling back to classical" % g["obstacle_method"])
```

- in `_tick`'s camera branch replace the single `detect_obstacles_camera` call:

```python
if self.use_cam_obs:
    obs_img = np.zeros(self._latest_img.shape[:2], bool)
    if self.obstacle_method in ("classical", "both") or self.yolo is None:
        obs_img |= obstacles.detect_obstacles_camera(self._latest_img, road)
    if self.yolo is not None:
        obs_img |= self.yolo.detect(self._latest_img)
    obst_grid |= bev.warp_to_bev(
        obs_img.astype(np.uint8) * 255, self._H, self.grid) > 127
```

- add the params to `config/perception_costmap.yaml` with a comment: *classical = no deps, yolo = accurate classes + Jetson .engine path, both = union.*

- [ ] **Step 5: Full suite green → commit**

```bash
git add perception_costmap/obstacles.py perception_costmap/costmap_node.py config/perception_costmap.yaml test/test_footprint.py
git commit -m "perception_costmap: YOLO detector class (load-once, footprint-strip projection)"
```

---

### Task 5: Segmenter factory + TwinLiteNet+ adapter (the road model, wired for real)

TwinLiteNet+ nano (34 K params, trained on BDD100K for drivable-area + lanes) is the "model that's good at roads". It goes behind the same callable interface as HSV so the node — and the eval tool — can switch by config.

**Files:**
- Modify: `perception_costmap/segmentation.py` (add `letterbox`, `HsvSegmenter`, `TwinLiteSegmenter`, `create_segmenter`; keep `segment_road` delegating for back-compat)
- Modify: `perception_costmap/costmap_node.py`, `config/perception_costmap.yaml`
- Test: `test/test_segmenter.py`

**Interfaces:**
- Produces: `create_segmenter(method, **kw) -> callable(img_bgr) -> bool mask`; `letterbox(img, new_size=640) -> (padded, ratio, (pad_left, pad_top))`. Task 8's eval tool consumes `create_segmenter`.
- Consumes: model-loading recipe from `perception/twinLiteNetTest.py` (repo clone + `nano.pth` weights — see its docstring for the one-time setup).

- [ ] **Step 1: Failing test (all offline — torch never imported by these paths)**

```python
# test/test_segmenter.py
import numpy as np
import pytest
from perception_costmap.segmentation import (
    create_segmenter, HsvSegmenter, letterbox)


def test_factory_hsv_returns_callable_mask():
    seg = create_segmenter("hsv", max_sat=60)
    img = np.full((40, 60, 3), (90, 90, 90), np.uint8)   # gray = "asphalt"
    mask = seg(img)
    assert mask.dtype == bool and mask.shape == (40, 60)
    assert mask.any()


def test_factory_unknown_method():
    with pytest.raises(ValueError):
        create_segmenter("segnet9000")


def test_letterbox_geometry():
    img = np.zeros((100, 200, 3), np.uint8)
    padded, ratio, (pl, pt) = letterbox(img, new_size=640)
    assert padded.shape[:2] == (640, 640)
    assert abs(ratio - 3.2) < 1e-6        # 640/200
    assert pl == 0 and pt == 160          # 100*3.2=320 tall -> 160 pad top
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement in `segmentation.py`** (move `letterbox` here verbatim from `perception/twinLiteNetTest.py`, credit its origin in the docstring):

```python
def letterbox(img, new_size=640, pad_color=(114, 114, 114)):
    """Aspect-preserving resize + gray padding to new_size x new_size.
    (Same scheme as perception/twinLiteNetTest.py / the YOLO family.)"""
    h, w = img.shape[:2]
    ratio = min(new_size / h, new_size / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_w, pad_h = (new_size - new_w) / 2, (new_size - new_h) / 2
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=pad_color)
    return padded, ratio, (left, top)


class HsvSegmenter:
    """Classical fallback — zero deps, calibrate HSV ranges per camera."""
    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, img_bgr):
        return segment_road_hsv(img_bgr, **self.kw)


class TwinLiteSegmenter:
    """
    TwinLiteNet+ drivable-area head as a road segmenter.

    Setup (once): clone https://github.com/chequanghuy/TwinLiteNetPlus and
    download nano.pth — full instructions in perception/twinLiteNetTest.py.
    Loads the network ONCE at construction; __call__ is inference only.
    """
    def __init__(self, repo_path, weights, config="nano", img_size=640,
                 device=None):
        import sys, argparse
        import torch                       # lazy: optional dependency
        self.torch = torch
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from model.model import TwinLiteNetPlus   # from the cloned repo
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = TwinLiteNetPlus(argparse.Namespace(config=config))
        model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model = model.to(self.device).eval()
        self.img_size = img_size

    def __call__(self, img_bgr):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        padded, _, (pl, pt) = letterbox(img_bgr, self.img_size)
        with torch.no_grad():
            t = torch.from_numpy(padded).to(self.device).float()
            t = t.permute(2, 0, 1).unsqueeze(0) / 255.0
            da_out, _ = self.model(t)          # (drivable-area, lanes)
        da = da_out[:, :, pt:self.img_size - pt, pl:self.img_size - pl]
        da = torch.nn.functional.interpolate(da, size=(h, w), mode="bilinear")
        return (torch.argmax(da, dim=1).squeeze(0).cpu().numpy() == 1)


def create_segmenter(method="hsv", **kw):
    """Factory: 'hsv' (classical, no deps) or 'twinlitenet' (learned,
    needs torch + cloned repo + weights — kw: repo_path, weights, config)."""
    if method == "hsv":
        return HsvSegmenter(**kw)
    if method == "twinlitenet":
        return TwinLiteSegmenter(**kw)
    raise ValueError("unknown segmentation method: %r" % (method,))
```

Update `segment_road` to delegate: `return create_segmenter(method, **kw)(img_bgr)` for `"hsv"`; for `"twinlitenet"` replace the `NotImplementedError` with the real dispatch (it will raise `ImportError`/`TypeError` naturally if torch/paths are missing — that's the honest failure now).

- [ ] **Step 4: Node wiring.** In `costmap_node.py.__init__`:
- new params: `("twinlite_repo_path", ""), ("twinlite_weights", ""), ("twinlite_config", "nano")`
- build the segmenter once (warm-load, like YOLO):

```python
try:
    if g["segmentation_method"] == "twinlitenet":
        self.segmenter = segmentation.create_segmenter(
            "twinlitenet", repo_path=g["twinlite_repo_path"],
            weights=g["twinlite_weights"], config=g["twinlite_config"])
    else:
        self.segmenter = segmentation.create_segmenter("hsv")
except Exception as e:                     # missing torch/weights/paths
    self.get_logger().warn("twinlitenet unavailable (%s); using hsv" % e)
    self.segmenter = segmentation.create_segmenter("hsv")
```

- in `_tick` replace `segmentation.segment_road(self._latest_img, method=self.seg_method)` with `self.segmenter(self._latest_img)`.
- config yaml: add the three twinlite params, and update the `segmentation_method` comment to `hsv | twinlitenet (learned, needs torch + weights)`.

- [ ] **Step 5: Full suite green → commit**

```bash
git add perception_costmap/segmentation.py perception_costmap/costmap_node.py config/perception_costmap.yaml test/test_segmenter.py
git commit -m "perception_costmap: segmenter factory + TwinLiteNet+ adapter (load-once)"
```

---

### Task 6: Multi-camera BEV fusion (the real car's 3-camera layout)

Each camera gets its own subscription, homography (now yaw-aware for side cameras), and FOV footprint. Fusion is trivially the BEV grid: OR the road masks, OR the obstacle masks, OR the known masks.

**Files:**
- Modify: `perception_costmap/bev.py` (add `homography_from_extrinsics`)
- Modify: `perception_costmap/costmap_node.py` (add `CameraSource`, loop cameras in `_tick`)
- Modify: `config/perception_costmap.yaml`
- Test: `test/test_geometry.py` (append)

**Interfaces:**
- Produces: `homography_from_extrinsics(K, cam_xyz, pitch_deg, yaw_deg, grid) -> 3x3 H` (image px → grid col,row). `homography_from_camera` becomes a thin wrapper: `homography_from_extrinsics(K, (0, 0, cam_height), pitch_deg, 0.0, grid)`.
- Config contract (consumed by Task 7's overlay tool): `cameras: [name, ...]`, then per name a nested block `name: {image_topic, camera_info_topic, ipm_mode, ipm_image_pts, ipm_world_pts, cam_x, cam_y, cam_height, cam_pitch_deg, cam_yaw_deg}` (rclpy flattens nested YAML to dotted params).

- [ ] **Step 1: Failing test — yaw geometry.** A camera yawed +90° (looking left) must see the same scene as a forward camera, rotated 90° in the world:

```python
# append to test/test_geometry.py
from perception_costmap.bev import homography_from_extrinsics

def _px_to_world(H, u, v, grid):
    p = H @ np.array([u, v, 1.0])
    col, row = p[0] / p[2], p[1] / p[2]
    return (grid.x_min + col * grid.resolution,
            grid.y_min + row * grid.resolution)


def test_yawed_camera_rotates_ground_points():
    g = GridSpec(x_min=-20, x_max=20, y_min=-20, y_max=20, resolution=0.1)
    K = np.array([[300.0, 0, 320], [0, 300.0, 180], [0, 0, 1]])
    H_fwd = homography_from_extrinsics(K, (0, 0, 1.6), 10.0, 0.0, g)
    H_left = homography_from_extrinsics(K, (0, 0, 1.6), 10.0, 90.0, g)
    u, v = 320.0, 260.0                     # a pixel below the horizon
    xf, yf = _px_to_world(H_fwd, u, v, g)
    xl, yl = _px_to_world(H_left, u, v, g)
    # rotating the camera +90 deg (left) maps (x, y) -> (-y, x)
    assert abs(xl - (-yf)) < 0.05 and abs(yl - xf) < 0.05
```

- [ ] **Step 2: Run — FAIL (no `homography_from_extrinsics`).**

- [ ] **Step 3: Implement in `bev.py`:**

```python
def homography_from_extrinsics(K, cam_xyz, pitch_deg, yaw_deg, grid: GridSpec) -> np.ndarray:
    """
    Ground-plane homography for a camera mounted at cam_xyz (robot frame,
    metres), pitched down pitch_deg and yawed yaw_deg (CCW, +left; 0 = facing
    +x). This is homography_from_camera generalised for side/rear cameras.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    th = np.radians(pitch_deg)
    yw = np.radians(yaw_deg)

    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(th), -np.sin(th)],
                   [0.0, np.sin(th), np.cos(th)]])
    Rz_inv = np.array([[np.cos(yw), np.sin(yw), 0.0],     # world -> cam-heading
                       [-np.sin(yw), np.cos(yw), 0.0],
                       [0.0, 0.0, 1.0]])
    R = Rx @ R0 @ Rz_inv

    C = np.asarray(cam_xyz, dtype=np.float64)
    t = -R @ C
    H_world2img = K @ np.column_stack((R[:, 0], R[:, 1], t))
    H_img2world = np.linalg.inv(H_world2img)
    return _world_to_grid_affine(grid) @ H_img2world
```

and shrink `homography_from_camera` to `return homography_from_extrinsics(K, (0.0, 0.0, cam_height), pitch_deg, 0.0, grid)` (keep its docstring; existing tests must stay green).

- [ ] **Step 4: Test passes. Refactor the node to `CameraSource`:**

Add to `costmap_node.py` (above the node class):

```python
class CameraSource:
    """One camera: its subscriptions, latest frame, homography and FOV mask.
    All parameters live under '<name>.' so a 3-camera car is 3 YAML blocks."""

    def __init__(self, node, name, grid):
        self.name, self.grid = name, grid
        d = lambda key, val: node.declare_parameter("%s.%s" % (name, key), val).value
        self.ipm_mode = d("ipm_mode", "points")
        self.image_pts = np.array(d("ipm_image_pts",
            [0.0, 160.0, 640.0, 160.0, 640.0, 320.0, 0.0, 320.0]), float).reshape(4, 2)
        self.world_pts = np.array(d("ipm_world_pts",
            [18.0, -8.0, 18.0, 8.0, 3.0, 4.0, 3.0, -4.0]), float).reshape(4, 2)
        self.cam_xyz = (d("cam_x", 0.0), d("cam_y", 0.0), d("cam_height", 1.6))
        self.pitch = d("cam_pitch_deg", 10.0)
        self.yaw = d("cam_yaw_deg", 0.0)
        self.img, self.stamp, self.K = None, 0.0, None
        self.H, self.known = None, None
        self._node = node
        node.create_subscription(Image, d("image_topic", "/camera/%s/image" % name),
                                 self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo,
                                 d("camera_info_topic", "/camera/%s/camera_info" % name),
                                 self._on_info, qos_profile_sensor_data)

    def _on_image(self, msg):
        if self._node._bridge is None:
            from cv_bridge import CvBridge
            self._node._bridge = CvBridge()
        self.img = self._node._bridge.imgmsg_to_cv2(msg, "bgr8")
        self.stamp = stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def ensure_homography(self):
        if self.H is not None:
            return True
        if self.ipm_mode == "camera":
            if self.K is None:
                return False
            self.H = bev.homography_from_extrinsics(
                self.K, self.cam_xyz, self.pitch, self.yaw, self.grid)
        else:
            self.H = bev.homography_from_points(
                self.image_pts, self.world_pts, self.grid)
        self.known = bev.bev_known_mask(self.H, self.img.shape, self.grid)
        return True
```

In `CostmapNode.__init__`: declare `("cameras", ["front"])`; after grid setup build `self.cameras = [CameraSource(self, n, self.grid) for n in g["cameras"]]`; delete the old single-camera params/subscriptions/`_ensure_homography`/`_on_image`/`_on_info` (keep `self._bridge = None`).

In `_tick`, the camera branch becomes a loop:

```python
known = np.zeros((self.grid.height, self.grid.width), bool)
saw_camera = False
for cam in self.cameras:
    if cam.img is None or not is_fresh(cam.stamp, now, self.img_stale):
        continue
    if not cam.ensure_homography():
        continue
    saw_camera = True
    road = self.segmenter(cam.img)
    road_bev |= bev.warp_to_bev(road.astype(np.uint8) * 255, cam.H, self.grid) > 127
    known |= cam.known
    if self.use_cam_obs:
        obs_img = ...   # exactly the Task 4 block, using cam.img / road / cam.H
        obst_grid |= bev.warp_to_bev(obs_img.astype(np.uint8) * 255, cam.H, self.grid) > 127
```

and the publish guard changes from `if known is None: return` to `if not saw_camera and not lidar_active: return`.

Config: replace the flat IPM block in `perception_costmap.yaml` with:

```yaml
    cameras: ["front"]           # real car: ["front", "left", "right"]
    front:
      image_topic: /camera/front/image
      camera_info_topic: /camera/front/camera_info
      ipm_mode: points           # points | camera
      ipm_image_pts: [0.0, 160.0, 640.0, 160.0, 640.0, 320.0, 0.0, 320.0]
      ipm_world_pts: [18.0, -8.0, 18.0, 8.0, 3.0, 4.0, 3.0, -4.0]
      cam_x: 0.0
      cam_y: 0.0
      cam_height: 1.6
      cam_pitch_deg: 10.0
      cam_yaw_deg: 0.0
    # left/right blocks: same keys, yaw +90 / -90, calibrated per camera
```

and `launch/perception.launch.py`: the remappings list is now wrong for multi-cam — remove the two hardcoded remappings and let topics come from the YAML blocks (keep `image_topic`/`lidar_topic` launch args remapping only `front.image_topic`-equivalent topics is messy; simplest correct move: keep the `lidar_topic` remap, drop the image remap, document "set camera topics in the YAML").

- [ ] **Step 5: Full suite green → commit**

```bash
git add perception_costmap/bev.py perception_costmap/costmap_node.py config/perception_costmap.yaml launch/perception.launch.py test/test_geometry.py
git commit -m "perception_costmap: multi-camera BEV fusion, yaw-aware homographies"
```

---

### Task 7: IPM calibration overlay tool (see your calibration before you trust it)

A bad homography poisons everything downstream and is invisible in RViz. This tool draws the metric grid (1 m lines) back onto a camera frame so a human can check "that line really is 5 m ahead" in seconds — in CARLA or standing in front of the real car.

**Files:**
- Create: `tools/ipm_overlay.py`
- Modify: `perception_costmap/bev.py` (add `draw_grid_on_image` — pure, testable)
- Test: `test/test_geometry.py` (append)

**Interfaces:**
- Produces: `bev.draw_grid_on_image(img_bgr, H, grid, spacing_m=1.0) -> img_bgr copy with grid lines + x-axis labels`.

- [ ] **Step 1: Failing test**

```python
# append to test/test_geometry.py
from perception_costmap.bev import draw_grid_on_image, homography_from_points

def test_draw_grid_overlay_changes_pixels_and_preserves_input():
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    img_pts = [(0, 200), (640, 200), (640, 360), (0, 360)]
    wld_pts = [(10, 5), (10, -5), (2, -2), (2, 2)]
    H = homography_from_points(img_pts, wld_pts, g)
    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_grid_on_image(img, H, g)
    assert out.shape == img.shape
    assert out.any()                 # lines were drawn
    assert not img.any()             # input untouched
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement in `bev.py`:**

```python
def draw_grid_on_image(img_bgr, H, grid: GridSpec, spacing_m=1.0):
    """
    Project the metric grid back into the camera image (green lines every
    spacing_m). Human calibration check: stand a marker at a known distance
    and confirm the drawn line lands on it. H maps image->grid, so we draw
    with its inverse.
    """
    out = img_bgr.copy()
    Hinv = np.linalg.inv(H)
    h, w = img_bgr.shape[:2]

    def world_to_px(x, y):
        col = (x - grid.x_min) / grid.resolution
        row = (y - grid.y_min) / grid.resolution
        p = Hinv @ np.array([col, row, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u, v = p[0] / p[2], p[1] / p[2]
        if -w <= u <= 2 * w and -h <= v <= 2 * h:
            return int(round(u)), int(round(v))
        return None

    for x in np.arange(np.ceil(grid.x_min), grid.x_max + 1e-6, spacing_m):
        pts = [world_to_px(x, y) for y in np.linspace(grid.y_min, grid.y_max, 40)]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 1)
        if pts:
            cv2.putText(out, "%gm" % x, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
    for y in np.arange(np.ceil(grid.y_min), grid.y_max + 1e-6, spacing_m):
        pts = [world_to_px(x, y) for x in np.linspace(max(0.5, grid.x_min), grid.x_max, 40)]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 1)
    return out
```

- [ ] **Step 4: Test passes. Write the CLI (`tools/ipm_overlay.py`):**

```python
#!/usr/bin/env python3
"""
Overlay the costmap's metric grid onto a camera frame to eyeball the IPM.

    python3 tools/ipm_overlay.py --image frame.png \
        --config config/perception_costmap.yaml --camera front --out overlay.png

Reads the same YAML the node uses, builds the same homography, draws 1 m grid
lines. If the 5 m line isn't 5 m away in the scene, fix the calibration
BEFORE debugging anything downstream. ROS not required.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.occupancy import GridSpec
from perception_costmap import bev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", default="front")
    ap.add_argument("--out", default="ipm_overlay.png")
    args = ap.parse_args()

    params = yaml.safe_load(open(args.config))["perception_costmap"]["ros__parameters"]
    grid = GridSpec(x_min=params["x_min"], x_max=params["x_max"],
                    y_min=params["y_min"], y_max=params["y_max"],
                    resolution=params["resolution"])
    cam = params[args.camera]
    img = cv2.imread(args.image)
    if img is None:
        sys.exit("could not read %s" % args.image)

    if cam.get("ipm_mode", "points") == "points":
        H = bev.homography_from_points(
            np.array(cam["ipm_image_pts"], float).reshape(4, 2),
            np.array(cam["ipm_world_pts"], float).reshape(4, 2), grid)
    else:
        sys.exit("camera mode needs a live camera_info; use ipm_mode: points here")

    cv2.imwrite(args.out, bev.draw_grid_on_image(img, H, grid))
    print("wrote %s -- check that the labelled distances match the scene" % args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke it on a synthetic frame, suite green, commit:**

```bash
python3 - <<'EOF'
import cv2, numpy as np
cv2.imwrite('/tmp/claude-frame.png', np.full((360, 640, 3), 80, np.uint8))
EOF
python3 tools/ipm_overlay.py --image /tmp/claude-frame.png \
    --config config/perception_costmap.yaml --camera front --out /tmp/claude-overlay.png
# Expected: "wrote /tmp/claude-overlay.png ..."
git add perception_costmap/bev.py tools/ipm_overlay.py test/test_geometry.py
git commit -m "perception_costmap: IPM grid-overlay calibration tool"
```

---

### Task 8: CARLA feed + measured accuracy (road-mask IoU vs ground truth)

Two tools for the 5090 box. `carla_feed.py` connects to CARLA 0.9.16, spawns an autopilot ego with front RGB + semantic-seg + lidar, publishes ROS2 topics the node consumes (bypassing the version-mismatched ros-bridge), and optionally dumps paired RGB/semantic PNGs. `eval_road_iou.py` scores segmenters against the semantic ground truth offline. The conversion math is pure and unit-tested here; the live scripts run only on the 5090 box.

**Files:**
- Create: `tools/carla_feed.py`, `tools/eval_road_iou.py`
- Create: `perception_costmap/carla_convert.py` (pure converters — importable by both tools and by tests)
- Test: `test/test_carla_convert.py`

**Interfaces:**
- Produces (in `carla_convert.py`): `bgra_bytes_to_bgr(raw, h, w) -> (h,w,3) uint8`; `carla_lidar_to_rep103(raw, sensor_z) -> (N,3) float`; `semantic_to_road_mask(sem_bgr, road_tags=(1, 24)) -> bool mask`; `mask_iou(a, b) -> float`.
- CRITICAL geometry facts encoded here: CARLA is **left-handed** (y right) — REP-103 needs `y = -y`; the semantic camera stores the tag in the **red channel** (index 2 of BGR); tag values changed at CARLA 0.9.13 — default road tags `(1, 24)` = Roads + RoadLines, but the tool prints the tags it actually sees so a wrong default is caught immediately.

- [ ] **Step 1: Failing test**

```python
# test/test_carla_convert.py
import numpy as np
from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103, semantic_to_road_mask, mask_iou)


def test_bgra_to_bgr():
    raw = np.array([[10, 20, 30, 255]], np.uint8).tobytes()   # 1x1 BGRA
    img = bgra_bytes_to_bgr(raw, 1, 1)
    assert img.shape == (1, 1, 3)
    assert list(img[0, 0]) == [10, 20, 30]


def test_lidar_flips_y_and_offsets_z():
    # CARLA lidar: (x, y_right, z, intensity) float32
    raw = np.array([[5.0, 2.0, 0.5, 0.9]], np.float32).tobytes()
    pts = carla_lidar_to_rep103(raw, sensor_z=1.8)
    assert pts.shape == (1, 3)
    assert pts[0, 0] == 5.0
    assert pts[0, 1] == -2.0          # left-handed -> REP-103
    assert abs(pts[0, 2] - 2.3) < 1e-6   # sensor frame -> base_link height


def test_semantic_road_mask_reads_red_channel():
    sem = np.zeros((2, 2, 3), np.uint8)
    sem[0, 0, 2] = 1      # Roads tag in R channel
    sem[1, 1, 2] = 24     # RoadLines
    m = semantic_to_road_mask(sem, road_tags=(1, 24))
    assert m[0, 0] and m[1, 1] and m.sum() == 2


def test_mask_iou():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [False, False]])
    assert abs(mask_iou(a, b) - 0.5) < 1e-9
    assert mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0
```

- [ ] **Step 2: Run — FAIL.**

- [ ] **Step 3: Implement `perception_costmap/carla_convert.py`:**

```python
"""
carla_convert.py — pure converters between CARLA sensor buffers and our
REP-103 / numpy conventions. No carla or ROS imports: these are the exact
functions where sim-to-real geometry bugs hide, so they are unit-tested.

Gotchas encoded here:
- CARLA uses a LEFT-handed frame (x fwd, y RIGHT, z up). REP-103 is y LEFT.
- The semantic camera writes the class tag into the red channel.
- Semantic tag ids changed in CARLA 0.9.13; verify with the printout in
  tools/carla_feed.py rather than trusting defaults blindly.
"""

import numpy as np


def bgra_bytes_to_bgr(raw, height, width):
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return arr[:, :, :3].copy()


def carla_lidar_to_rep103(raw, sensor_z):
    """CARLA lidar buffer (x, y_right, z, intensity) -> (N,3) base_link
    points: flip y for handedness, add mount height so z is above ground."""
    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
    pts = pts.astype(np.float64)
    pts[:, 1] *= -1.0
    pts[:, 2] += float(sensor_z)
    return pts


def semantic_to_road_mask(sem_bgr, road_tags=(1, 24)):
    tags = sem_bgr[:, :, 2]
    mask = np.zeros(tags.shape, dtype=bool)
    for t in road_tags:
        mask |= (tags == t)
    return mask


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum()) / float(union)
```

- [ ] **Step 4: Tests pass. Write `tools/carla_feed.py`:**

```python
#!/usr/bin/env python3
"""
CARLA 0.9.16 -> ROS2 feed for perception_costmap. Runs on the x86/5090 box
(CARLA has no ARM build; the Jetson never sees this file).

Spawns an autopilot ego with front RGB + semantic camera + lidar and
publishes:
    /camera/front/image        sensor_msgs/Image (bgr8)
    /camera/front/camera_info  sensor_msgs/CameraInfo (K from FOV)
    /lidar/points              sensor_msgs/PointCloud2 (REP-103, base_link)

--dump-dir writes paired frames (NNNN_rgb.png / NNNN_sem.png) every
--dump-every ticks for tools/eval_road_iou.py.

    python3 tools/carla_feed.py --host 127.0.0.1 --dump-dir /tmp/pairs
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103)

W, H, FOV = 640, 360, 90.0
CAM_X, CAM_Z, CAM_PITCH = 1.5, 1.6, -10.0     # CARLA pitch: negative = down
LIDAR_Z = 1.8


def make_camera_info(stamp, frame):
    from sensor_msgs.msg import CameraInfo
    fx = W / (2.0 * math.tan(math.radians(FOV) / 2.0))
    msg = CameraInfo()
    msg.header.stamp, msg.header.frame_id = stamp, frame
    msg.width, msg.height = W, H
    msg.k = [fx, 0.0, W / 2.0, 0.0, fx, H / 2.0, 0.0, 0.0, 1.0]
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--dump-dir", default=None)
    ap.add_argument("--dump-every", type=int, default=20)
    args = ap.parse_args()

    import carla
    import cv2
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Header

    rclpy.init()
    node = rclpy.create_node("carla_feed")
    pub_img = node.create_publisher(Image, "/camera/front/image", qos_profile_sensor_data)
    pub_info = node.create_publisher(CameraInfo, "/camera/front/camera_info", qos_profile_sensor_data)
    pub_pts = node.create_publisher(PointCloud2, "/lidar/points", qos_profile_sensor_data)

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    bp = world.get_blueprint_library()

    ego = world.spawn_actor(bp.filter("vehicle.tesla.model3")[0],
                            world.get_map().get_spawn_points()[0])
    ego.set_autopilot(True)

    def cam_bp(kind):
        b = bp.find(kind)
        b.set_attribute("image_size_x", str(W))
        b.set_attribute("image_size_y", str(H))
        b.set_attribute("fov", str(FOV))
        return b

    cam_tf = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z),
                             carla.Rotation(pitch=CAM_PITCH))
    rgb = world.spawn_actor(cam_bp("sensor.camera.rgb"), cam_tf, attach_to=ego)
    sem = world.spawn_actor(cam_bp("sensor.camera.semantic_segmentation"),
                            cam_tf, attach_to=ego)
    lidar_bp = bp.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", "30")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar = world.spawn_actor(lidar_bp,
                              carla.Transform(carla.Location(z=LIDAR_Z)),
                              attach_to=ego)

    state = {"rgb": None, "sem": None, "n": 0}
    dump = Path(args.dump_dir) if args.dump_dir else None
    if dump:
        dump.mkdir(parents=True, exist_ok=True)

    def now():
        return node.get_clock().now().to_msg()

    def on_rgb(image):
        img = bgra_bytes_to_bgr(bytes(image.raw_data), H, W)
        state["rgb"] = img
        stamp = now()
        msg = Image()
        msg.header.stamp, msg.header.frame_id = stamp, "camera_front"
        msg.height, msg.width = H, W
        msg.encoding, msg.step = "bgr8", W * 3
        msg.data = img.tobytes()
        pub_img.publish(msg)
        pub_info.publish(make_camera_info(stamp, "camera_front"))

    def on_sem(image):
        state["sem"] = bgra_bytes_to_bgr(bytes(image.raw_data), H, W)
        state["n"] += 1
        if dump and state["rgb"] is not None and state["n"] % args.dump_every == 0:
            i = state["n"] // args.dump_every
            cv2.imwrite(str(dump / ("%04d_rgb.png" % i)), state["rgb"])
            cv2.imwrite(str(dump / ("%04d_sem.png" % i)), state["sem"])
            tags = sorted(np.unique(state["sem"][:, :, 2]).tolist())
            print("pair %04d saved; semantic tags present: %s" % (i, tags))

    def on_lidar(meas):
        pts = carla_lidar_to_rep103(bytes(meas.raw_data), LIDAR_Z)
        hdr = Header()
        hdr.stamp, hdr.frame_id = now(), "base_link"
        pub_pts.publish(point_cloud2.create_cloud_xyz32(hdr, pts.tolist()))

    rgb.listen(on_rgb)
    sem.listen(on_sem)
    lidar.listen(on_lidar)
    print("feeding ROS2. camera K published from FOV; IPM 'camera' mode: "
          "cam_height=%.2f pitch=%.1f. Ctrl-C to stop." % (CAM_Z, -CAM_PITCH))
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for a in (rgb, sem, lidar, ego):
            a.destroy()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write `tools/eval_road_iou.py`:**

```python
#!/usr/bin/env python3
"""
Score road segmenters against CARLA semantic ground truth. Offline: feed it
the paired PNGs from carla_feed.py --dump-dir. Prints per-method mean IoU in
image space (does the mask match the road?) and reports the winner.

    python3 tools/eval_road_iou.py --pairs /tmp/pairs \
        [--twinlite-repo TwinLiteNetPlus --twinlite-weights nano.pth]
    # add --road-tags if the printed tag list from carla_feed says 1/24 is wrong
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.carla_convert import semantic_to_road_mask, mask_iou
from perception_costmap.segmentation import create_segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--road-tags", type=int, nargs="+", default=[1, 24])
    ap.add_argument("--twinlite-repo", default=None)
    ap.add_argument("--twinlite-weights", default=None)
    args = ap.parse_args()

    segmenters = {"hsv": create_segmenter("hsv")}
    if args.twinlite_repo and args.twinlite_weights:
        try:
            segmenters["twinlitenet"] = create_segmenter(
                "twinlitenet", repo_path=args.twinlite_repo,
                weights=args.twinlite_weights)
        except Exception as e:
            print("twinlitenet unavailable: %s" % e)

    pairs = sorted(Path(args.pairs).glob("*_rgb.png"))
    if not pairs:
        sys.exit("no *_rgb.png in %s" % args.pairs)

    scores = {name: [] for name in segmenters}
    for rgb_path in pairs:
        sem_path = Path(str(rgb_path).replace("_rgb.png", "_sem.png"))
        rgb, sem = cv2.imread(str(rgb_path)), cv2.imread(str(sem_path))
        if rgb is None or sem is None:
            continue
        truth = semantic_to_road_mask(sem, tuple(args.road_tags))
        for name, seg in segmenters.items():
            scores[name].append(mask_iou(seg(rgb), truth))

    print("\nroad-mask IoU vs CARLA semantic truth (%d frames):" % len(pairs))
    for name, vals in sorted(scores.items()):
        print("  %-12s mean %.3f   min %.3f" % (name, np.mean(vals), np.min(vals)))
    best = max(scores, key=lambda n: np.mean(scores[n]))
    print("winner: %s -> set segmentation_method: %s in the YAML" % (best, best))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Full suite green → commit**

```bash
git add perception_costmap/carla_convert.py tools/carla_feed.py tools/eval_road_iou.py test/test_carla_convert.py
git commit -m "perception_costmap: CARLA feed + road-IoU eval (measured accuracy)"
```

**Run procedure on the 5090 box (avl@100.113.89.64), for whoever executes the smoke test:** start CARLA; `python3 tools/carla_feed.py --dump-dir /tmp/pairs`; in a second shell `ros2 launch perception_costmap perception.launch.py rviz:=true` with `front.ipm_mode: camera` (K/height/pitch printed by the feed are exact); confirm road=free / cars=lethal in RViz tracking the autopilot; then `python3 tools/eval_road_iou.py --pairs /tmp/pairs` and record the IoU table in the PR.

---

### Task 9: Jetson deployment path — TensorRT export, benchmark, DEPLOY.md

**Files:**
- Create: `tools/export_trt.py`, `tools/bench_perception.py`, `DEPLOY.md`

**Interfaces:**
- Consumes: `YoloObstacleDetector(weights=...)` accepting `.engine` (Task 4), `create_segmenter` (Task 5).

- [ ] **Step 1: `tools/export_trt.py`** (run ON the Jetson — engines are hardware-specific):

```python
#!/usr/bin/env python3
"""
Export YOLOv8 to a TensorRT FP16 engine. RUN THIS ON THE JETSON — a .engine
built on the 5090 will not load on the Orin. Afterwards set
yolo_weights: /path/yolov8n.engine in the YAML; the detector class loads
either format.

    python3 tools/export_trt.py --weights yolov8n.pt --imgsz 640
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO
    path = YOLO(args.weights).export(format="engine", half=True, imgsz=args.imgsz)
    print("engine written: %s" % path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: `tools/bench_perception.py`** — same pipeline the node runs, timed per stage, no ROS needed, runs identically on laptop / 5090 / Jetson:

```python
#!/usr/bin/env python3
"""
Per-stage timing of the perception pipeline on synthetic frames (or --image).
Run on any machine; the number that matters is the Jetson's.

    python3 tools/bench_perception.py --frames 50
    python3 tools/bench_perception.py --frames 50 --yolo-weights yolov8n.engine \
        --twinlite-repo TwinLiteNetPlus --twinlite-weights nano.pth
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from perception_costmap.occupancy import GridSpec, build_cost_array
from perception_costmap import bev, obstacles
from perception_costmap.segmentation import create_segmenter


def timed(fn, frames):
    t0 = time.perf_counter()
    for f in frames:
        out = fn(f)
    dt = (time.perf_counter() - t0) / len(frames)
    return out, dt * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--image", default=None)
    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--twinlite-repo", default=None)
    ap.add_argument("--twinlite-weights", default=None)
    args = ap.parse_args()

    if args.image:
        base = cv2.imread(args.image)
    else:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (360, 640, 3), np.uint8).astype(np.uint8)
        base[200:, :] = (90, 90, 90)      # bottom half "asphalt"
    frames = [base.copy() for _ in range(args.frames)]

    grid = GridSpec()
    H = bev.homography_from_points(
        [(0, 200), (640, 200), (640, 360), (0, 360)],
        [(16, 10), (16, -10), (3, -3), (3, 3)], grid)

    rows = []
    seg = create_segmenter("hsv")
    road, ms = timed(seg, frames)
    rows.append(("segment (hsv)", ms))

    if args.twinlite_repo and args.twinlite_weights:
        tl = create_segmenter("twinlitenet", repo_path=args.twinlite_repo,
                              weights=args.twinlite_weights)
        _, ms = timed(tl, frames)
        rows.append(("segment (twinlitenet)", ms))

    _, ms = timed(lambda f: obstacles.detect_obstacles_camera(f, road), frames)
    rows.append(("obstacles (classical)", ms))

    if args.yolo_weights:
        det = obstacles.YoloObstacleDetector(weights=args.yolo_weights)
        _, ms = timed(det.detect, frames)
        rows.append(("obstacles (yolo)", ms))

    road_u8 = road.astype(np.uint8) * 255
    _, ms = timed(lambda f: bev.warp_to_bev(road_u8, H, grid), frames)
    rows.append(("bev warp", ms))

    road_bev = bev.warp_to_bev(road_u8, H, grid) > 127
    obst = np.zeros_like(road_bev)
    _, ms = timed(lambda f: build_cost_array(grid, road_bev, obst), frames)
    rows.append(("build cost array", ms))

    total = sum(ms for _, ms in rows)
    print("\n%-24s %8s" % ("stage", "ms/frame"))
    for name, ms in rows:
        print("%-24s %8.2f" % (name, ms))
    print("%-24s %8.2f  (~%.1f Hz worst-case serial)" % ("TOTAL", total, 1000.0 / total))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the benchmark locally — expect a table and no crash:**

Run: `python3 tools/bench_perception.py --frames 20`
Expected: stage table ending in `TOTAL ... Hz` (hsv pipeline should be well under 50 ms/frame on the laptop).

- [ ] **Step 4: Write `DEPLOY.md`** — the Jetson Orin Nano bring-up checklist:

```markdown
# Deploying to the car computer (Jetson Orin Nano, "dinosaur")

The Jetson runs the identical perception stack against real sensors. CARLA
never runs here (x86 only) — `tools/carla_feed.py` is replaced by real camera
and lidar drivers publishing the same topics.

## 0. Facts that decide everything
- Kernel 5.15.148-tegra => JetPack 5.x => Ubuntu 20.04 => native ROS2 is
  **Foxy**. Our target is Humble. Two options:
  a) (recommended) run the stack in a Humble container:
     `dustynv/ros:humble-desktop-l4t-r35.4.1` with `--runtime nvidia`, or
  b) build on Foxy natively — this package avoids Humble-only APIs, but Nav2
     Foxy is EOL; prefer (a).
- 8 GB RAM shared CPU/GPU. Add swap before building: 
  `sudo fallocate -l 8G /swap && sudo mkswap /swap && sudo swapon /swap`
- Power: `sudo nvpmodel -m 0 && sudo jetson_clocks` (MAXN) before benchmarks.

## 1. Torch/ultralytics (inside the container or JetPack env)
- NEVER `pip install torch` — that pulls a CPU wheel. Use NVIDIA's Jetson
  wheel matching the JetPack version (developer.nvidia.com/embedded → PyTorch
  for Jetson), then `pip install ultralytics --no-deps` + its light deps.

## 2. Build + verify (10 min)
    cd ros2_ws && colcon build --packages-select perception_costmap
    source install/setup.bash
    cd src/perception_costmap && PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q
    python3 tools/bench_perception.py --frames 50          # hsv baseline

## 3. Models
    python3 tools/export_trt.py --weights yolov8n.pt       # ON the Jetson
    python3 tools/bench_perception.py --frames 50 --yolo-weights yolov8n.engine
    # target: yolo stage <= 25 ms (≈2x realtime headroom at 10 Hz with seg)
    # TwinLiteNet nano: benchmark with --twinlite-*; if too slow on CPU fall
    # back to hsv until a TensorRT export of it is done.

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
- Nav2 local costmap (config/nav2_costmap_params.yaml) mirrors it.
```

- [ ] **Step 5: Suite green → commit**

```bash
git add tools/export_trt.py tools/bench_perception.py DEPLOY.md
git commit -m "perception_costmap: TensorRT export, pipeline benchmark, Jetson deploy guide"
```

---

### Task 10: Documentation sync + final validation

**Files:**
- Modify: `README.md` (Status section, tools section), `DESIGN.md` (dataflow: segmenter factory → multi-cam BEV fusion → temporal filter → OccupancyGrid)

- [ ] **Step 1:** Update `README.md`: add a `## Tools` table (ipm_overlay / carla_feed / eval_road_iou / export_trt / bench_perception, one line each); rewrite `## Status` to list what is now true (QoS-correct, multi-camera, YOLO + TwinLiteNet wired behind config, temporal filtering, measured-IoU workflow) and what still needs a human/hardware (per-camera calibration values, on-Jetson TRT export, real sensor drivers). Update `DESIGN.md`'s pipeline diagram to include the temporal filter and camera list. Keep the tone plain — no metrics we haven't measured.

- [ ] **Step 2: Final gate, then push:**

```bash
PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q          # expect ~30 tests, 0 failures
cd ~/carla-nav2-backup/ros2_ws && colcon build --packages-select perception_costmap  # if colcon available locally
git add README.md DESIGN.md
git commit -m "perception_costmap: docs sync for perception v2"
git push origin feature/alexander
```

---

## Execution order & dependencies

```
Task 1 (QoS/staleness) ─┬─> Task 3 (temporal, uses is_fresh)
Task 2 (vectorize)      │
Task 4 (YOLO)  ─────────┼─> Task 6 (multi-cam consumes Tasks 1,4,5 node shape)
Task 5 (segmenter) ─────┘         └─> Task 7 (overlay uses config contract)
Task 8 (CARLA/eval) needs Task 5 (factory); live half runs only on the 5090 box
Task 9 (Jetson) needs Tasks 4+5; export/bench run on-device
Task 10 last
```

Tasks 1–7 and the testable halves of 8–9 complete on this laptop. The two external checkpoints (record actual numbers, don't skip): **(A)** 5090 box — carla_feed + RViz + IoU table; **(B)** Jetson — colcon build, pytest, bench table with `.engine`.
