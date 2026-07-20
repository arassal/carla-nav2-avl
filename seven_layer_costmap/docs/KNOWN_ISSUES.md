# Known issues and acceptance blockers

This file separates implemented behavior from claims that require the Ubuntu/ZED
target or physical measurements.

## Camera calibration is provisional

`config/camera_mounts.yaml` was transcribed from an inch-based sketch. The sketch
uses a lateral-axis sign convention that conflicts with ROS REP-103, the front
camera Z sign is questionable, and no measured roll or pitch was supplied.

The pipeline now applies full translation plus roll/pitch/yaw. Replace every
`mounts.<camera>.translation` and `mounts.<camera>.rpy` value in
`config/seven_layer_costmap.yaml` after calibrating the physical rig. Until then,
double edges, tilted ground, and displaced obstacles are expected and are not a
software synchronization failure.

Acceptance check: display `/seven_layer_costmap/points/fused` in RViz and inspect
the same planar wall/ground patch where camera fields overlap. It should form one
surface rather than three displaced surfaces.

## Three concurrent SVO decoders need target validation

Stereolabs documents one SVO per wrapper launch. This project runs three isolated
wrapper instances. A prior target attempt was made while three live ZED stacks,
Velodyne, RViz, and the old perception stack were still consuming resources, so
it did not prove concurrent playback. Stop conflicting camera processes before
testing. If three `NEURAL` depth instances exceed GPU capacity, use
`zed_svo_realtime_override.yaml` (`NEURAL_LIGHT`) or reduce publication rate and
resolution. Quality mode intentionally runs slower than wall clock rather than
dropping frames.

## Recording synchronization cannot be invented later

The synchronizer rejects a camera set whose corrected SVO timestamps differ by
more than `max_camera_skew_s`. Constant offsets can be corrected through
`timestamp_offsets_s`; changing drift, unrelated recordings, and dropped-frame
patterns cannot. Diagnostics publish current, mean, and maximum skew plus the
violation count.

## Ground filtering depends on extrinsics

The dependency-light ground remover finds a dominant horizontal height band. It
is suitable for wiring and visualization after mount calibration, but it is not a
road-grade plane/terrain estimator. Uneven ground, steep slopes, and incorrect
camera pitch can create false obstacles. A calibrated RANSAC/semantic ground
model should replace it if those scenes matter.

## Visual semantics are baselines

The depth-derived BEV and point cloud are the primary trustworthy outputs. The
lane, static/dynamic, and prediction layers are deterministic integration
baselines, not trained or safety-validated perception models. Motion prediction
and temporal memory are disabled in the vision-only default. The former
traffic-regulation and road-condition heuristics were removed from the runtime
contract in version 0.8.

## Local validation boundary

Dependency-light algorithms, configuration contracts, Python compilation, and
YAML parsing can be tested on Windows. ROS 2 Humble, RViz, three ZED wrapper
instances, GPU depth inference, and real SVO2 playback require the Ubuntu/Jetson
target. Do not report end-to-end success until the target acceptance checklist in
`docs/REPRODUCIBILITY.md` has been completed.
