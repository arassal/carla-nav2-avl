# perception_costmap

Camera + lidar perception that publishes a **Nav2-compatible costmap**: where
the road is, and where the obstacles are. See [DESIGN.md](DESIGN.md) for the
architecture.

## Outputs

| Topic | Type | Meaning |
|-------|------|---------|
| `/perception/costmap` | `nav_msgs/OccupancyGrid` | road = 0, off-road/obstacle = 100, unseen = -1 |
| `/perception/obstacle_points` | `sensor_msgs/PointCloud2` | lidar obstacle returns (for Nav2's obstacle layer) |

## Build

```bash
cd ros2_ws
colcon build --packages-select perception_costmap
source install/setup.bash
```

## Run

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
PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q     # 39 offline tests
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
| `tools/export_trt.py` | Exports YOLOv8 `.pt` weights to a TensorRT `.engine`. Must be run ON the Jetson — an engine built on the 5090 will not load on the Orin. |
| `tools/bench_perception.py` | Per-stage timing (segmentation, obstacle detection, BEV warp, cost-array build) on synthetic frames, with warm-up excluded and a TOTAL that reflects only the stages a deployed config actually runs serially (one segmenter + one obstacle method, not every combination benchmarked). |

## Status

**Done and verified offline + under ROS2 (this laptop, 39 tests green):**
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

**Still needs a human or the actual hardware:**
- Real per-camera IPM calibration values — the YAML ships with placeholder
  `ipm_image_pts`/`ipm_world_pts`/mounting numbers; calibrate each camera with
  `tools/ipm_overlay.py` against a tape measure (or a CARLA frame first).
- On-Jetson TensorRT export (`tools/export_trt.py`) and on-Jetson
  `bench_perception.py` numbers (the laptop numbers above do not transfer).
- Real camera/lidar ROS2 drivers on the Jetson (currently only CARLA topics
  via `carla_feed.py` are exercised).
- A live CARLA smoke test + `eval_road_iou.py` IoU table run on the 5090 box
  (RViz visual check + measured hsv-vs-twinlitenet numbers).
- TwinLiteNet+ weights download and a working torch environment to actually
  exercise that segmentation path (the adapter code is wired and covered by
  tests with a stub, but the real model has not been run end-to-end here).
