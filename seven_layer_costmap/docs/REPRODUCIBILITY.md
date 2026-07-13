# Reproducing the seven-layer three-ZED-X costmap

This document describes the camera-only milestone that is actually present in
`feature/jchy05`. It does not claim vehicle-safety validation.

## Input and sensor contract

The real playback launch accepts exactly three ZED X SVO/SVO2 files: front,
left, and right. The ZED SDK derives these ROS inputs from the recordings:

- rectified left RGB image from each camera;
- registered stereo depth from each camera;
- camera calibration from each camera; and
- visual-inertial odometry from the front camera.

There is no LiDAR, Velodyne, `LaserScan`, radar, or external `PointCloud2` input.
The perception code back-projects stereo-depth pixels internally. A ZED wrapper
may advertise its optional camera-derived point-cloud topic, but this package has
no subscriber for it; that topic is neither LiDAR nor a required input.

## Reference environment

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10
- Stereolabs ZED SDK with ZED X/SVO2 support (the sample files were written by
  SDK 5.2.0)
- `zed-ros2-wrapper`, `colcon`, Nav2, RViz2, `cv_bridge`, NumPy, OpenCV, and
  PyYAML
- NVIDIA/Jetson hardware supported by the selected ZED SDK

Run `scripts/check_environment.sh` after sourcing ROS and the ZED workspace. It
reports missing commands and ROS packages without modifying the machine.

## Obtain and verify the sample recordings

The three five-second recordings are in this
[Google Drive folder](https://drive.google.com/drive/folders/1Ew1lBB8XXJfox14D_zE0RPyZoE5vuR9j).
Their serial numbers, byte sizes, and SHA-256 hashes are recorded in
`config/sample_svo_manifest.yaml`.

On Ubuntu, verify downloaded copies with:

```bash
sha256sum front_5s.svo2 left_5s.svo2 right_5s.svo2
ZED_SVO_Editor -inf front_5s.svo2
ZED_SVO_Editor -inf left_5s.svo2
ZED_SVO_Editor -inf right_5s.svo2
```

Each file should report 75 frames at 15 Hz, 960 x 600, and the serial number in
the manifest.

## Clone, build, and test

```bash
git clone --branch feature/jchy05 --single-branch \
  https://github.com/arassal/carla-nav2-avl.git carla-nav2-avl-jchy05
cd carla-nav2-avl-jchy05/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --base-paths src ../seven_layer_costmap \
  --packages-select seven_layer_costmap
source install/setup.bash
colcon test --packages-select seven_layer_costmap
colcon test-result --verbose
```

For a hardware-free integration check:

```bash
ros2 launch seven_layer_costmap offline_pipeline.launch.py
```

In other terminals, source the same setup and inspect:

```bash
ros2 topic echo --once /seven_layer_costmap/perception_status
ros2 topic hz /seven_layer_costmap/costmap
ros2 launch seven_layer_costmap visualize.launch.py
```

## Replay three real recordings

Do not run playback with live wrappers using the same `zed_front`, `zed_left`,
and `zed_right` names. On a stationary test machine, first shut down conflicting
camera launches through their normal supervisor, then run:

```bash
ros2 launch seven_layer_costmap three_svo_costmap.launch.py \
  front_svo:=/absolute/path/front_5s.svo2 \
  left_svo:=/absolute/path/left_5s.svo2 \
  right_svo:=/absolute/path/right_5s.svo2
```

The launch rejects missing or relative paths. A three-camera set is accepted
only when RGB/depth pairs are fresh and all corrected timestamps satisfy the
configured skew limits.

## Blind-spot behavior

The two configurable wedges between the front and side cameras are not treated
as automatically lethal. Valid stereo-depth rays create an internal observed
mask. Within each blind wedge:

- directly observed clear space uses `blind_spot_clear_cost` (default 0);
- unobserved space uses `blind_spot_unknown_cost` (default 25);
- static/temporal obstacle memory, predicted motion, and inflation can raise the
  cost up to 100.

The mask is an internal input to the existing lanelet, voxel, prediction, and
inflation behavior; it does not add an eighth published layer. Tune the wedge
angles and ranges only after calibrating the physical camera fields of view.

## What remains machine- and vehicle-specific

- Camera translations and rotations are provisional sketch conversions.
- Timestamp offsets must be measured for each recording set.
- The lanelet layer is a local vision-derived corridor, not a Lanelet2 HD map.
- Traffic, road-condition, and object-prediction baselines require trained and
  validated replacements for production use.
- A clean end-to-end acceptance run with the sample SVO2 files must still be
  recorded after ensuring playback does not conflict with live vehicle wrappers.

## Validation recorded on 2026-07-13

- 38 dependency-light tests passed on Windows and on the target Jetson.
- ROS 2 Humble built package version 0.6.0 on the target Jetson.
- `verification.launch.py` published all seven layers and the fused map.
- `offline_pipeline.launch.py` reported `ACTIVE` through the same RGB, depth,
  calibration, odometry, and visibility callbacks used by SVO playback.
- `ZED_SVO_Editor` validated all three sample recordings against the manifest.
- A single front recording opened in ZED SDK 5.2.0 and advertised the expected
  RGB, registered-depth, calibration, and odometry topics through completion.

A concurrent three-SVO acceptance result is intentionally not claimed. During
the attempted run, the vehicle machine was already operating three live ZED SDK
instances, the older perception stack, Velodyne drivers, and RViz. The additional
three playback instances did not all load under that resource contention. Repeat
the acceptance run on a stationary vehicle after stopping the conflicting live
camera stack through its normal supervisor; ROS-domain isolation prevents topic
collisions but does not free GPU/CPU resources.
