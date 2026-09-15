# perception_costmap

Camera + lidar perception that publishes a **Nav2-compatible costmap**: where
the road is, and where the obstacles are. See [DESIGN.md](DESIGN.md) for the
architecture.

## Outputs

| Topic | Type | Meaning |
|-------|------|---------|
| `/perception/costmap` | `nav_msgs/OccupancyGrid` | road = 0, caution = 1-96, off-road = 97, lethal = 100 |
| `/perception/known` | `nav_msgs/OccupancyGrid` | authoritative observed-area mask for visualization |
| `/perception/obstacle_points` | `sensor_msgs/PointCloud2` | lidar obstacle returns (for Nav2's obstacle layer) |
| `/perception/costmap_cloud` | `sensor_msgs/PointCloud2` | nearest lethal/off-road boundary per bearing consumed by Nav2 |

## Services

| Service | Type | Meaning |
|---------|------|---------|
| `/perception/reset` | `std_srvs/Trigger` | Drop all accumulated state — per-cell temporal confidence, motion-compensation reference, buffered samples, counters — without restarting. Models, parameters and homographies are untouched, so the node publishes again on the next tick. |

Call it between runs with `deploy/fresh_run.sh`, which also clears both Nav2
costmaps and reports whether the stack is genuinely clean.

**Why it matters for IGVC:** each run must carry nothing over from the last.
Nothing here is written to disk, both Nav2 costmaps are rolling with no static
layer, and STVL decays in ~3 s — the temporal filter is the only state that
outlives a run, and it does so for the life of the process. Restarting the
whole stack cleared it by accident; this makes it deliberate and instant.

## Build

```bash
cd ros2_ws
colcon build --packages-select perception_costmap
source install/setup.bash
```

## Run

**On the car**, start only what perception needs. The launch reuses anything
already running, so two people can't start a camera twice:

```bash
ros2 launch perception_costmap perception_stack.launch.py                # sensors + 3 cameras + costmap
ros2 launch perception_costmap perception_stack.launch.py cameras:=front # one camera
ros2 launch perception_costmap perception_stack.launch.py --show-args    # everything else
```

Cameras use the lean `config/zed_perception_*.yaml` profiles: RGB, depth and
confidence only, no point cloud or positional tracking. It does not start viz,
streaming or RViz. See ISSUES.md P1-P4 for why.

**Just the node** (CARLA, or with sensors already up):

```bash
# defaults (topics in config/perception_costmap.yaml)
ros2 launch perception_costmap perception.launch.py

# point it at CARLA / real sensor topics: lidar is still a launch arg,
# camera topics are NOT (see below) -- edit the YAML for those
ros2 launch perception_costmap perception.launch.py \
    lidar_topic:=/carla/ego/lidar \
    rviz:=true
```

Camera topics are no longer launch args. With multi-camera BEV fusion, each
camera gets its own YAML block under `cameras: [...]` in
`config/perception_costmap.yaml` (`image_topic`, `camera_info_topic`, IPM
mode/points, mounting). Point a camera at a different topic by editing that
block (or passing `config:=/path/to/override.yaml`). `lidar_topic` remains a
launch arg since there's only ever one lidar.

## Feed it into Nav2

`config/nav2_costmap_params.yaml` stacks our outputs as costmap layers
(`static_layer` <- the OccupancyGrid, `obstacle_layer` <- the lidar points,
plus inflation). Load it onto your Nav2 costmap nodes / bringup.

## Calibrate the IPM (do this once per camera)

The bird's-eye projection needs to know how image pixels map to the ground.
Two options in `config/perception_costmap.yaml`:

- `ipm_mode: points` — set `ipm_image_pts` (4 pixels) and `ipm_world_pts`
  (their ground positions in metres, x forward / y left). Easiest: pick a flat
  rectangle on the ground in one frame and measure it.
- `ipm_mode: camera` — derive it from `camera_info` K + `cam_height` /
  `cam_pitch_deg`. Convenient in CARLA where these are exact; verify against a
  real frame.

## Tests

```bash
cd ros2_ws/src/perception_costmap
PYTHONPATH=. python3 -m pytest test -q     # 90 passed (2026-09-14)
```

## CARLA smoke test (on the x86 / 5090 box)

1. Start CARLA, then run `tools/carla_feed.py` to spawn an autopilot ego with
   front RGB + lidar and publish `/camera/front/image` + `camera_info` +
   `/lidar/points` (see the Tools table below).
2. `ros2 launch perception_costmap perception.launch.py rviz:=true` (edit the
   `cameras:` block in the YAML first if you changed camera topics/mounting).
3. In RViz add the `/perception/costmap` OccupancyGrid display. You should see
   the road as free (green-ish), the off-road and any vehicles as lethal.
4. Drive the ego; the costmap should track the road ahead and mark obstacles.
5. Then load `nav2_costmap_params.yaml` into Nav2 and confirm the local
   costmap reflects the same road/obstacles.

## Tools

| Tool | What it does |
|------|--------------|
| `tools/ipm_overlay.py` | Draws the node's 1 m grid lines onto a camera frame using the YAML's per-camera homography, so you can eyeball whether the IPM calibration is right before debugging anything downstream. |
| `tools/carla_feed.py` | Runs on the x86/5090 box (CARLA has no ARM build): spawns an autopilot ego in CARLA 0.9.16, publishes front camera + camera_info + lidar as ROS2 topics, and can dump paired RGB/semantic frames for `eval_road_iou.py`. Prints `cam_x`/`cam_height`/`cam_pitch_deg` for the YAML. |
| `tools/eval_road_iou.py` | Scores `hsv` vs `twinlitenet` road-mask IoU against CARLA semantic ground truth from `carla_feed.py --dump-dir` pairs; reports per-method mean IoU and the winner. |
| `tools/eval_ipm_calibration.py` | Reports metric RMSE, mean, p95, and maximum camera-to-ground projection error from surveyed point correspondences. |
| `tools/benchmark_models.py` | Measures HSV, TwinLiteNet TensorRT, and YOLO latency on representative images without changing the live configuration. |
| `tools/measure_zed_sync.py` | Measures RGB/depth/confidence timestamp alignment for all three ZED X cameras. |
| `tools/analyze_zed_depth.py` | Reports valid depth coverage and confidence-threshold retention from live synchronized ZED frames. |
| `deploy/record_perception_bag.sh` | Records the full ZED/perception/odometry/TF contract for deterministic regression evidence. |
| `tools/export_trt.py` | Exports YOLOv8 `.pt` weights to a TensorRT `.engine`. Must be run ON the Jetson — an engine built on the 5090 will not load on the Orin. |
| `tools/bench_perception.py` | Per-stage timing (segmentation, obstacle detection, BEV warp, cost-array build) on synthetic frames, with warm-up excluded and a TOTAL that reflects only the stages a deployed config actually runs serially (one segmenter + one obstacle method, not every combination benchmarked). |

## Status

### Dinosaur accuracy upgrade (2026-09-03)

Detector inference runs on a single bounded worker. The ROS timer keeps only
the newest pending camera set instead of accumulating stale frames, and each
result is odometry-reprojected from its capture pose before temporal fusion.
When a required detector is configured, publication waits for its first valid
result rather than briefly publishing a road-only map during model warm-up.

- The front camera receives obstacle inference every costmap cycle. One side
  camera is processed every other cycle in round-robin order, closing the
  previous side-obstacle gap while sustaining a measured 10.0 Hz costmap rate.
- Registered ZED depth now places detected obstacle pixels directly in the
  metric robot-frame grid. Missing, stale, or invalid depth explicitly falls
  back to the existing ground-plane IPM projection.
- ZED SDK confidence maps are timestamp-matched to RGB and depth. Projection
  rejects low-confidence pixels, detection-edge bleed, and robust depth
  outliers before transforming points into the robot frame.
- Debug rendering is decoupled from perception: the costmap remains 10 Hz,
  while its 40,000-point RViz color cloud and software-rendered view run at
  2 Hz to preserve CPU for cameras, inference, localization, and control.
- Temporal confidence is retained separately for person, vehicle, cone, and
  generic obstacles and is reprojected using wheel odometry as the robot moves.
- Semantic safety zones survive the Nav2 cloud bridge: confirmed objects are
  lethal, with hard exclusion radii of 1.2 m for people, 0.6 m for vehicles,
  0.5 m for generic obstacles, and 0.2 m for cones. People and vehicles mark
  after one confident result; generic noise still requires two-frame evidence.
- The operator RViz view is separate from the machine map: a muted
  blue/amber/red ramp, neutral checkerboard for unobserved space, the exact
  Nav2 collision boundary, and vehicle pose are shown without changing costs.
- `eval_road_iou.py` supports binary campus labels and emits IoU, precision,
  recall, F1, and latency JSON. Model promotion requires labeled campus results;
  the presence of a TensorRT engine alone is not an acceptance criterion.
- Blind-region policy is intentionally unchanged by this upgrade.

**Done and verified offline + on the Dinosaur Jetson (offline suite green):**
- Sensor-data (`BEST_EFFORT`) QoS on every subscription, with `image_stale_sec`
  / `lidar_stale_sec` guards that drop frames instead of building a costmap
  from stale data.
- Vectorized lidar point binning; floor-vs-int boundary semantics at the grid
  edge pinned by a regression test.
- Temporal obstacle confidence filter (`temporal.py`): per-cell confidence
  rises by `temporal_hit` on detection, decays by `temporal_miss` when
  observed-empty, reports lethal at `>= temporal_threshold`. "Observed" is the
  fused camera FOV, or the whole grid when lidar is active.
- YOLOv8 obstacle detector (`obstacle_method: classical|yolo|both`): loads
  once, rasterizes a footprint strip (bottom fraction of each box) onto the
  grid, accepts `.pt` or TensorRT `.engine` weights, warm-loads at startup and
  falls back to classical on any load failure.
- Segmenter factory (`segmentation_method: hsv|twinlitenet`) with a
  TwinLiteNet+ adapter (repo/weights/config params); loads once, crops output
  to content extent, warm-loads with a fallback to `hsv` on any failure.
- Multi-camera BEV fusion: `cameras: [...]` list plus a nested YAML block per
  camera (topics, IPM mode/points, `cam_x/y/height/pitch/yaw`); yaw-aware
  `homography_from_extrinsics`; per-tick fusion ORs road/known/obstacles
  across cameras; lidar-only ticks still publish (grid `UNKNOWN` except lidar
  obstacles).
- Tooling: `ipm_overlay.py`, `carla_feed.py`, `eval_road_iou.py`,
  `export_trt.py`, `bench_perception.py` (see Tools table above) and
  `DEPLOY.md` for the Jetson bring-up sequence.
- Laptop bench (HSV + classical obstacles, no YOLO/TwinLiteNet): ~4 ms/frame
  total for segment + obstacles + warp + cost-array build. This is a laptop
  number for sanity-checking the pipeline shape, not a Jetson number — see
  `DEPLOY.md` for the on-device bench.

**Still needs field data:**
- Surveyed campus points for each camera. Run `eval_ipm_calibration.py` and
  require <=0.25 m RMSE before accepting any extrinsic adjustment.
- Pixel-labeled campus frames covering sun, shadow, glare, wet pavement, grass,
  curbs, cones, people, and vehicles. Run `eval_road_iou.py`; do not promote
  TwinLiteNet over HSV until measured IoU/F1 improves without unacceptable
  latency or false-drivable errors.
- A stationary supervised obstacle-layout check, followed by a low-speed closed
  course test. Software tests and indoor live topics cannot establish physical
  stopping distance or outdoor segmentation accuracy.
