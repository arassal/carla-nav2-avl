# Seven-layer active costmap — milestone 1

This ROS 2 Humble package lives at repository root alongside `docs`,
`image_thresholding`, `ros2_ws`, and `scripts`. It is an isolated CARLA-first,
three-ZED-X/SVO-ready prototype. It publishes one active fused `nav_msgs/OccupancyGrid` at
`/seven_layer_costmap/costmap` and keeps all seven contributing grids visible at
`/seven_layer_costmap/layers/<layer_name>`.

## Three-SVO pipeline

The production-intent launch accepts exactly three absolute SVO paths and starts
one namespaced ZED wrapper for each recording:

```bash
ros2 launch seven_layer_costmap three_svo_costmap.launch.py \
  front_svo:=/data/front.svo2 \
  left_svo:=/data/left.svo2 \
  right_svo:=/data/right.svo2
```

The wrappers publish rectified RGB, registered metric depth, and camera
calibration. `three_zed_perception` accepts a set only when all three cameras have
advanced and their SVO timestamps are within `max_camera_skew_s` (50 ms by
default). A missing, stale, or misaligned camera stops the six perception-layer
updates; the fail-closed fusion node then stops the master map after its stale
timeout. Status is available on `/seven_layer_costmap/perception_status`.

Stereolabs documents SVO replay as a single-file wrapper operation, so this launch
uses three wrapper instances. It deliberately disables their TF/map publication;
the vehicle rig transforms are owned by this project and must be calibrated.

The front wrapper's visual-inertial odometry is required. Observations are stored
as sparse world-frame voxels and projected back into a rolling vehicle-centered
grid. This prevents stationary obstacles from smearing when the vehicle moves or
turns. Sampled depth rays clear previously occupied free space, while unobserved
voxels expire after `voxel_persistence_s`. A configurable rectangular ego mask
rejects visible vehicle-body returns before occupancy processing.

## Layer contract

| Layer | Purpose | Milestone input | Vehicle input required later |
|---|---|---|---|
| lanelet | penalize locally non-drivable space | front-image marking estimate and conservative corridor | trained lane/drivable segmentation for production |
| static obstacle | persistent geometry | repeated fused depth occupancy | semantic static/dynamic classification |
| spatio-temporal voxel | recent 3-D occupancy with decay | fused ZED depth with 2 s persistence | tune marking/ray-clearing on SVO data |
| prediction | future dynamic occupancy | depth-component centroid tracking and constant velocity | multi-object semantic tracker and lane-aware predictor |
| inflation | vehicle clearance | union of static, voxel, and prediction layers | validated vehicle footprint and speed-dependent margins |
| traffic regulation | temporary stop restrictions | conservative red-pixel signal gate | trained traffic-light/sign/road-marking detector |
| road condition | traction/visibility penalty | three-camera RGB heuristic | trained and safety-validated road/weather model |

The fusion node uses weighted maximum cost. With the default fail-closed setting,
it does not publish if any of the seven layers is absent, stale, or uses different
grid geometry. This catches broken producers instead of silently dropping safety
information.

## Build and verify on Ubuntu 22.04 / ROS 2 Humble

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
# The package is one directory above this workspace, so include both base paths.
colcon build --symlink-install \
  --base-paths src ../seven_layer_costmap \
  --packages-select seven_layer_costmap
source install/setup.bash
colcon test --packages-select seven_layer_costmap
colcon test-result --verbose
ros2 launch seven_layer_costmap verification.launch.py
ros2 topic hz /seven_layer_costmap/costmap
ros2 topic echo --once /seven_layer_costmap/costmap
```

Alternatively, create a development symlink at
`ros2_ws/src/seven_layer_costmap` pointing to the repository-root package. Do not
copy it into the workspace, because maintaining two package copies invites drift.

`verification.launch.py` publishes all seven deterministic layers and requires no
CARLA or ZED hardware. `three_svo_costmap.launch.py` is the real input path.
`milestone_demo.launch.py` remains a CARLA/live-image wiring harness.

For a stronger hardware-free check, run the actual perception pipeline against
synthetic ZED-compatible RGB, registered depth, calibration, and odometry topics:

```bash
ros2 launch seven_layer_costmap offline_pipeline.launch.py
ros2 topic echo /seven_layer_costmap/perception_status
ros2 topic hz /seven_layer_costmap/costmap
```

Unlike `verification.launch.py`, this does not publish prebuilt layer grids. It
drives the same image/depth/odometry callbacks used by three-SVO playback.

`config/nav2_consumer.yaml` provides an experimental Nav2 Humble `StaticLayer`
consumer for the fused topic and a provisional rectangular vehicle footprint.
This integration cannot be runtime-validated in the current Windows workspace;
verify frame transforms, rolling-map behavior, lifecycle transitions, and planner
response on the target Ubuntu/ROS installation before enabling vehicle control.

## CARLA milestone procedure

1. Spawn three `sensor.camera.rgb` actors at the provisional transforms in
   `config/camera_mounts.yaml`; spawn matching depth cameras for obstacle work.
2. Name/remap their ROS topics to the three configured ZED-compatible image topics.
3. Start CARLA with native ROS 2 and simulation time.
4. Run `milestone_demo.launch.py`.
5. Exercise dry, rain, night/fog, stopped actor, moving actor, traffic light, and
   lane-boundary scenarios while recording every individual layer and the master.

The image classifier is a deterministic baseline for wiring and CARLA scenario
verification. It uses lower-image brightness, contrast, and channel spread to
classify `dry`, `wet`, `low_visibility`, or `snow_or_glare`. It is not a learned
model and must not be treated as vehicle-safe perception.

## What can and cannot be inferred from only three SVO files

Camera intrinsics and synchronized stereo depth are available through each ZED
wrapper. This package derives a **local, rolling perception costmap** from those
data. With no Lanelet2/OpenDRIVE/GNSS input, it cannot create a globally referenced
HD-map lanelet layer or recover regulations that are not visible in the videos.
The layer named `lanelet` is therefore a local vision-derived drivable-lane proxy.
The traffic layer handles visible red signals only in the dependency-free
baseline. Signs, arrows, speed limits, yield rules, and occluded signals require a
trained detector and suitable labeled data.

The baseline algorithms intentionally avoid downloading model weights at runtime.
They make the complete dataflow executable and testable, but lane estimation,
red-light detection, road conditions, object identity, and prediction are not
vehicle-ready until replaced or validated with appropriate models and recordings.

## Calibration assumptions requiring confirmation

The supplied sketch used inches and stated left-negative Y. Values were multiplied
by exactly `0.0254`. This conflicts with standard ROS REP-103 (`+Y` left), so the
file preserves the sketch values and labels them provisional. The forward camera's
written `z=-4.25 in` also appears inconsistent with its drawn placement. No camera
roll/pitch/yaw measurements were supplied; nominal outward-facing yaw values are
simulation placeholders. Do not deploy these transforms on a vehicle until they
are measured and validated with TF/point-cloud overlays.

## Preflight and acceptance checks

- All seven topics update at the requested rate with identical geometry/frame.
- The master stops within one stale timeout when any producer is stopped.
- Synthetic obstacles, predictions, stop line, lane boundary, inflation, and road
  penalty appear in both the relevant debug layer and the master.
- Unit tests pass, followed by a ROS launch smoke test on Humble.
- CARLA scenario evidence is recorded before claiming integration verification.

Before accepting an SVO run, also verify:

- All three paths are absolute and each wrapper reaches playback state.
- `/seven_layer_costmap/perception_status` stays `ACTIVE` with acceptable skew.
- Registered depth uses `32FC1` meters and each `CameraInfo` is nonzero.
- Point-cloud overlays align after replacing provisional mount transforms.
- ZED odometry remains synchronized with image/depth timestamps and does not jump.
- The ego exclusion rectangle masks only vehicle bodywork, not nearby obstacles.
- A newly visible clear ray removes a disappeared obstacle without waiting for expiry.
- Stopping any one wrapper causes the fused map to stop within one second.
- CPU/GPU load sustains the configured publication rate without skipped frames.

No `.svo` recordings are required for unit and synthetic integration testing. Real
three-SVO playback, perception accuracy measurement, calibration, and vehicle
deployment are explicitly separate acceptance gates.
