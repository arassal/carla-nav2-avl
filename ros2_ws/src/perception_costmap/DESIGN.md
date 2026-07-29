# Perception → Costmap → Nav2

Design for the perception stack that turns camera + lidar into a Nav2-usable
costmap: where the road is, and where the obstacles are.

## Goal

Produce a 2D costmap, in the robot frame, where:

- **drivable road** = free (cost 0)
- **off-road** (grass, sidewalk, anything not road) = high cost / lethal
- **obstacles** (vehicles, pedestrians, cones) = lethal
- **unknown** (not yet observed) = unknown (-1)

Nav2's planner + controller then plan paths that stay on the road and avoid
obstacles. We do **not** reinvent the planner or the costmap engine — we feed
Nav2's existing `nav2_costmap_2d` through standard interfaces.

## How perception feeds Nav2 (the key decision)

Nav2 builds its costmap from stacked *layers*. We use three:

| Layer | Source | What it does |
|-------|--------|--------------|
| `StaticLayer` | our `/perception/costmap` (`nav_msgs/OccupancyGrid`) | road vs off-road + camera-seen obstacles |
| `obstacle_layer` | our `/perception/obstacle_points` (`sensor_msgs/PointCloud2`) | lidar obstacles, with raytrace clearing |
| `inflation_layer` | (built-in) | inflate lethal cells by the car's radius |

Two reasons for splitting it this way:

1. **Road/drivable area** is naturally a dense grid (every cell is road or
   not), so an `OccupancyGrid` is the right message and `StaticLayer` ingests
   it directly — no custom C++ plugin needed.
2. **Obstacles** are sparse and benefit from Nav2's native raytrace *clearing*
   (a cell stops being an obstacle once the sensor sees through it). That is
   exactly what `obstacle_layer` does when fed a `PointCloud2`. Lidar points
   are already metric, so this path is the most reliable.

This keeps everything on standard ROS2 messages, which is also what makes it
portable to the Jetson later (swap the sim sensor topics for real driver
topics; the perception node and Nav2 config are unchanged).

## Pipeline

Per camera in `cameras: [...]` (e.g. `front`, and on the car `left`/`right`):

```
camera Image ─┐
              ├─► segmentation.py (factory: hsv | twinlitenet) ─► road mask ─┐
CameraInfo  ──┘                                                              │
                                              bev.py (per-camera,            ├─► fused BEV grid
camera Image ─► obstacles.py (classical|yolo|both) ─► obstacle mask ────────┘   (road / known / obstacles,
                                              yaw-aware homography)              OR'd across cameras)

lidar PointCloud2 ─► obstacles.py (ground filter + cluster) ──► obstacle_points ─┘   |
                                                                    (PointCloud2)     ▼
                                                                          temporal.py (per-cell
                                                                          confidence filter)
                                                                                       │
                                                                                       ▼
                                                                    occupancy.py ─► /perception/costmap
                                                                                    (OccupancyGrid)
```

- **segmentation.py** — a factory selects the road-mask backend by
  `segmentation_method`: `hsv` (threshold + largest blob, reused from Adam
  Castillo's `perception/costmap.py`) or `twinlitenet` (a TwinLiteNet+
  adapter, loaded once, output cropped to content extent). The learned path
  warm-loads at startup and falls back to `hsv` on any load failure, so a
  missing model/weights never takes the node down.
- **obstacles.py** — obstacle mask from the camera (`obstacle_method`:
  `classical` contrast, `yolo` — a YOLOv8 detector loaded once that
  rasterizes a footprint strip per detection box, or `both` unioned) and
  obstacle points from lidar (remove ground plane by z-band, keep what's
  above it). YOLO also warm-loads with a fallback to `classical`.
- **bev.py** — inverse-perspective mapping: per camera, warp its
  segmentation/obstacle mask from image space into the shared top-down metric
  grid using that camera's homography. The homography comes from either 4
  point correspondences (`ipm_mode: points`) or camera intrinsics + yaw-aware
  extrinsics (`ipm_mode: camera`, `homography_from_extrinsics`,
  `cam_x/y/height/pitch/yaw`); calibratable from a single frame. In CARLA we
  know these exactly; on the car we calibrate once per camera with
  `tools/ipm_overlay.py`.
- **Multi-camera fusion** (in `costmap_node.py`) — each configured camera
  contributes independently; the fused BEV grid is the OR of road / known /
  obstacle cells across all cameras plus lidar. If lidar is the only fresh
  sensor on a given tick (all cameras stale), the grid still publishes with
  everything `UNKNOWN` except lidar-detected obstacles, rather than skipping
  the tick.
- **temporal.py** — a per-cell confidence filter smooths obstacle flicker
  across ticks: confidence rises by `temporal_hit` when a cell is detected as
  an obstacle, decays by `temporal_miss` when the cell is *observed* (inside
  the fused camera FOV, or the whole grid when lidar is active) but empty,
  and the cell reports lethal only once confidence crosses
  `temporal_threshold`. Disabled entirely via `temporal_enabled: false`.
- **occupancy.py** — fuse masks into a `nav_msgs/OccupancyGrid` with correct
  metadata (resolution, origin, frame). This is the offline-testable heart.
- **costmap_node.py** — the ROS2 node wiring subscriptions (per-camera +
  lidar, sensor-data QoS, staleness guards) → modules → publishers, all
  parameterized.

## Grid geometry

- Robot-centric, REP-103 axes: **+x forward, +y left**.
- Default: 20 m forward × 20 m wide, **0.1 m/cell** → 200 × 200 cells.
- Robot at the bottom-center of the forward range.
- `OccupancyGrid.info.origin` is set so cell→world matches the chosen extent;
  `header.frame_id = base_link` (or `ego`), republished every frame.
- Cost encoding follows ROS convention: `0` free, `100` lethal, `-1` unknown,
  intermediate values for "off-road but traversable if forced."

## Topics & frames

| Topic | Type | Dir |
|-------|------|-----|
| `<camera>/image_topic` (one per name in `cameras: [...]`, e.g. `/camera/front/image`) | `sensor_msgs/Image` | in |
| `<camera>/camera_info_topic` (e.g. `/camera/front/camera_info`) | `sensor_msgs/CameraInfo` | in |
| `/lidar/points` (remappable via the `lidar_topic` launch arg) | `sensor_msgs/PointCloud2` | in (optional) |
| `/perception/costmap` | `nav_msgs/OccupancyGrid` | out |
| `/perception/obstacle_points` | `sensor_msgs/PointCloud2` | out |

Camera topics are all-in-YAML, not launch args — see the per-camera blocks in
`config/perception_costmap.yaml`. Only `lidar_topic` is a launch arg (there's
one lidar).

Frames: `map → odom → base_link → camera/lidar`. Perception publishes in
`base_link`; Nav2 handles the rest via TF.

## Sim vs car (sim-to-real)

| | Dev / CARLA (x86) | Car / Jetson (arm64) |
|---|---|---|
| camera/lidar source | CARLA sensors | real camera + lidar drivers |
| perception node | same code | same code |
| Nav2 config | same | same |
| control output | `cmd_to_carla` | real actuator/CAN node |

Only the sensor source and the actuator sink change. The perception → costmap
→ Nav2 core is identical, which is the whole point of doing it on standard
messages.

## Build order

1. ✅ this design
2. scaffold `perception_costmap` package
3. geometry core (`bev.py`, `occupancy.py`) + offline test
4. `segmentation.py`, `obstacles.py`
5. `costmap_node.py`
6. Nav2 costmap params + launch
7. validation (offline test + colcon build, then CARLA smoke test on the 5090)

## Calibration / known TODOs

- Every camera's IPM homography needs real numbers from CameraInfo + mounting
  (`cam_x/y/height/pitch/yaw` or 4 point correspondences); the YAML ships
  with placeholder values per camera until calibrated with
  `tools/ipm_overlay.py`.
- HSV road thresholds are lighting-dependent; the `twinlitenet` seg path
  exists to remove that fragility, but hasn't been run end-to-end against
  real weights here yet.
- Lidar ground-plane removal assumes roughly flat ground near the car.
- `temporal_hit` / `temporal_miss` / `temporal_threshold` were chosen to be
  reasonable, not measured against real obstacle flicker rates; revisit once
  there's a real camera feed to tune against.
- TensorRT export (`tools/export_trt.py`) and the perception bench
  (`tools/bench_perception.py`) need to be re-run on the Jetson itself —
  numbers from this laptop don't transfer (different CPU/GPU, no TensorRT).
