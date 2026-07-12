# Seven-layer active costmap — milestone 1

This ROS 2 Humble package is an isolated CARLA-first, three-ZED-X/SVO-ready
prototype. It publishes one active fused `nav_msgs/OccupancyGrid` at
`/seven_layer_costmap/costmap` and keeps all seven contributing grids visible at
`/seven_layer_costmap/layers/<layer_name>`.

## Layer contract

| Layer | Purpose | Milestone input | Vehicle input required later |
|---|---|---|---|
| lanelet | penalize off-lane space | deterministic test corridor | Lanelet2/OpenDRIVE rasterizer |
| static obstacle | persistent mapped geometry | deterministic test wall | map/static perception |
| spatio-temporal voxel | recent 3-D occupancy with decay | deterministic test obstacle; core voxel algorithm included | fused ZED depth point clouds |
| prediction | future dynamic occupancy | constant-velocity test track; core rasterizer included | timestamped tracked objects |
| inflation | vehicle clearance | generated from test obstacles | Nav2 or included inflation algorithm |
| traffic regulation | stop lines/no-entry/speed restrictions | deterministic stop line | CARLA/Lanelet2 signals and rules |
| road condition | traction/visibility penalty | CARLA/ZED RGB heuristic | trained and safety-validated camera model |

The fusion node uses weighted maximum cost. With the default fail-closed setting,
it does not publish if any of the seven layers is absent, stale, or uses different
grid geometry. This catches broken producers instead of silently dropping safety
information.

## Build and verify on Ubuntu 22.04 / ROS 2 Humble

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select seven_layer_costmap
source install/setup.bash
colcon test --packages-select seven_layer_costmap
colcon test-result --verbose
ros2 launch seven_layer_costmap verification.launch.py
ros2 topic hz /seven_layer_costmap/costmap
ros2 topic echo --once /seven_layer_costmap/costmap
```

`verification.launch.py` publishes all seven deterministic layers and requires no
CARLA or ZED hardware. `milestone_demo.launch.py` replaces only the road-condition
test layer with live images. Start CARLA camera publishers first, or remap the
three image topics in `config/seven_layer_costmap.yaml`.

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

## Future SVO integration

Run one namespaced `zed_wrapper` instance per SVO and remap its RGB/depth/camera
info output to the same three topic prefixes. The costmap code should not need to
change. Playback must use a shared timestamp/clock; if recordings are not hardware
synchronized, align them before fusion and reject frames exceeding the configured
skew. Camera intrinsics come from each SVO; extrinsics come from
`camera_mounts.yaml` after physical validation.

## Calibration assumptions requiring confirmation

The supplied sketch used inches and stated left-negative Y. Values were multiplied
by exactly `0.0254`. This conflicts with standard ROS REP-103 (`+Y` left), so the
file preserves the sketch values and labels them provisional. The forward camera's
written `z=-4.25 in` also appears inconsistent with its drawn placement. No camera
roll/pitch/yaw measurements were supplied; nominal outward-facing yaw values are
simulation placeholders. Do not deploy these transforms on a vehicle until they
are measured and validated with TF/point-cloud overlays.

## Definition of milestone-one completion

- All seven topics update at the requested rate with identical geometry/frame.
- The master stops within one stale timeout when any producer is stopped.
- Synthetic obstacles, predictions, stop line, lane boundary, inflation, and road
  penalty appear in both the relevant debug layer and the master.
- Unit tests pass, followed by a ROS launch smoke test on Humble.
- CARLA scenario evidence is recorded before claiming integration verification.

No `.svo` recordings are required for this milestone. Real SVO playback, learned
road-condition inference, and vehicle deployment are explicitly later gates.
