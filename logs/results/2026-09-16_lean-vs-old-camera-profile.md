# Lean vs old ZED profile, measured on the car

**Date:** 2026-09-16 · **Car:** dinosaur (Jetson AGX Orin, Humble, MAXN, ZED SDK 5.2.0)
**Code:** `~/chris_test/carla-nav2-avl` = `copy` @ `0fedca3` + PRs #1-#4

**Front camera excluded, and why:** during these runs it would not open --
`CAMERA STREAM FAILED TO START`, with nvargus-daemon reporting
`AlreadyAllocated: Device 0 (of 1) is in use`. **Another program on the car (a
teammate streaming that camera) held it open**; the ZED X allows one client at
a time. Not a hardware fault: once that program stopped, the front camera came
straight back at 8.5 Hz with no reboot and no daemon restart. Both runs
therefore use **left + right only**, through
`perception_leftright.yaml` (a copy of `perception_dinosaur.yaml` with
`cameras: [left, right]`, `required_cameras: [left]`).

## Method

The only difference between the two runs is the camera YAML. Same costmap
node (same build, same config), same two cameras, same sensors, no RViz, no
viz_node, no streaming. 30 s sample each; per-process CPU read from `/proc`
over the window (100% = one core); `tegrastats` at 1 s.

- **OLD:** `zed_camera.launch.py` per camera with avros_bringup's
  `zed_left.yaml` / `zed_right.yaml`, plus
  `perception_stack.launch.py cameras:=none`.
- **LEAN:** `perception_stack.launch.py cameras:=left,right`, which uses
  `config/zed_perception_{left,right}.yaml`.

## Results

| | OLD configs | LEAN profile | change |
|---|---|---|---|
| RGB / depth publish rate | 15.0 / 15.1 Hz | 7.9 / 8.1 Hz | capped at 8 by design |
| ZED topics per camera | 27 | **23** | point cloud + odom/pose gone |
| point_cloud topics | 2 | **0** | |
| odom/pose topics | 6 | **0** | positional tracking off |
| zed_left driver CPU | 26% | **15%** | |
| zed_right driver CPU | 24% | **16%** | |
| **both camera drivers** | **50%** | **31%** | **-38%** |
| costmap_node CPU | 122% | **108%** | -14 points (fewer frames to convert) |
| CPU, 12-core mean | 24.8% | **22.4%** | -2.4 points (~0.3 core) |
| GPU (GR3D) mean / max | 59.2% / 99% | 56.9% / 99% | -2.3 points |
| RAM | 6096 MB | **5918 MB** | -178 MB |
| /perception/costmap | 9.64 Hz | **9.99 Hz** | steadier |

## Reading the numbers

- The camera drivers cost **about a third less CPU**, and the savings come
  from three things the lean profile turns off or caps: publishing at 8 Hz
  instead of 15, no point cloud, and no positional tracking. Most of it is the
  frame rate; perception ticks at 10 Hz and runs YOLO on a side camera every
  0.4 s, so 15 Hz was work nothing consumed.
- The costmap node also drops 14 points, because it converts fewer images
  through cv_bridge.
- GPU barely moves. Depth is still computed per grab, and the detectors
  dominate GPU time.
- Freed CPU is roughly a third of a core in total on this two-camera setup.
  With three cameras it should be about half a core.
- `/perception/costmap` stayed at 10 Hz in both runs; this is a load
  reduction, not a throughput change.

## Three cameras, old profile (measured separately, no disruption)

The boot stack *is* the old profile, so it could be measured while running,
with all three cameras up (13:21, same 30 s method):

| | three cameras, old configs |
|---|---|
| camera drivers | front 26% + left 28% + right 26% = **80% of a core** |
| costmap_node | 108% |
| viz_node / costmap_rgb / rviz2 | 33% / 11% / 3% |
| CPU, 12-core mean | 36.0% |
| GPU mean | 57.6% |
| RAM | 7362 MB |
| topics | 27 per camera, 3 point clouds, 9 odom/pose |
| /perception/costmap | 10.01 Hz |

**The front camera costs 26% while publishing at 8 Hz, the same as the side
cameras at 15 Hz.** So a camera's cost is dominated by the point cloud and
positional tracking, not the publish rate.

## Three cameras, lean profile (14:10, 10-minute window)

`perception_stack.launch.py` with all three cameras, same 30 s method:

| three cameras | old configs | **lean** | change |
|---|---|---|---|
| camera drivers | 26 + 28 + 26 = **80%** | 17 + 16 + 16 = **49%** | **-39%** |
| costmap_node | 108% | 110% | unchanged |
| CPU, 12-core mean | 36.0% | **27.3%** | -8.7 points |
| GPU mean | 57.6% | 57.3% | unchanged |
| RAM | 7362 MB | **6991 MB** | -371 MB |
| /perception/costmap | 10.01 Hz | **10.11 Hz** | unchanged |

**Read the CPU row carefully.** The camera drivers really do drop 31 points
(~0.31 core). The rest of the 8.7-point system difference is that the lean
launch does not start viz_node, costmap_rgb and RViz, which together cost
~47% (0.47 core) in the old run. That is the launch's "only what perception
needs" choice, not a camera-config effect, and those viewers can be started
separately whenever an operator wants them.

The earlier extrapolation from the two-camera run (45-48%) matched the
measured 49%.

## Caveats

- The lean run used two cameras; the three-camera old profile was measured
  separately (above), and the three-camera lean figure is an extrapolation.
- One 30 s sample per configuration, stationary robot, indoors.
- `depth_stabilization: 0` in the lean profile was not evaluated for quality
  here; that needs a look at the depth image with something in view.
- The costmap node's own load (ISSUES.md P2) is untouched by this change.

## Also found

- `ros2 launch` rejects an empty argument value, so `cameras:=''` fails with
  "malformed launch argument". `cameras:=none` is now accepted for a
  sensors-and-costmap-only run.
- **`AlreadyAllocated: Device 0 (of 1) is in use` means another process owns
  that camera**, not that the hardware is broken. The wrapper's watchdog
  relaunch loop can never fix it -- it just retries every 5 s (55 times in one
  case) while saying only `CAMERA STREAM FAILED TO START`. Worth teaching a
  recovery script to read the nvargus error and name the holder instead.
