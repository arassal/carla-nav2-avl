# Perception profile, ZED driver audit, lean launch test

**Date:** 2026-09-14 · **Host:** laptop (x86, ROS 2 Jazzy, apt OpenCV 4.6) — not the car
**Branch:** `feature/christian-lean-launch` (profiled together with PR #3's modules)

## 1. Costmap node profile

Car config (`perception_dinosaur.yaml`) with the TensorRT detectors replaced
by `obstacle_method: classical` and no depth topics. A fixture published 3
cameras at the car's current rates (front 8 Hz, left/right 15 Hz), 960x600
BGR, plus `/wheel_odom` at 50 Hz. `python3 -m cProfile` on the node for ~25 s.

    /perception/costmap     9.92 Hz
    node CPU                165% of one core (incl. cProfile overhead), 31 threads
    _tick                   265 calls, 38.2 ms each (budget 100 ms at 10 Hz)

| work (main thread, per tick) | ms | share of _tick |
|---|---|---|
| HSV road segmentation, 3 cams (722 calls) | 21.5 | 56% |
|   of which `np.isin` on the label image | 3.2 | 8% |
| white-line mask, 3 cams | 5.2 | 14% |
| `build_cost_array` (`cv2.inpaint` ~2 ms) | 4.5 | 12% |
| `reproject_grid` (detector results + temporal filters) | 3.8 | 10% |

Top self-time: `cvtColor` 1.83 s, `split` 0.95 s, `ndarray.astype` 0.77 s,
rclpy `_take_subscription` 0.72 s, `np.in1d` 0.71 s, `reproject_grid` 0.62 s.

Not measured: the detector worker thread (profiled main thread only), YOLO /
cone TensorRT on the Jetson GPU, the ZED drivers themselves.

## 2. ZED driver audit (zed-ros2-wrapper v5.2.2 source)

- `depth.publish_depth_confidence` defaults to `false`; the
  `confidence/confidence_map` publisher is only created when it is true
  (`zed_camera_component_video_depth.cpp`). avros_bringup's zed_*.yaml on
  GitHub never set it, while `perception_dinosaur.yaml` subscribes to it.
- `isDepthRequired()` returns true when any depth topic has subscribers *or*
  positional tracking needs depth, so with `pos_tracking_enabled: true` depth
  runs on every grab with zero subscribers.
- Consumers of ZED topics in this repo: RGB image + camera_info (perception,
  viz_node, 3 RViz configs, dashboard), depth + confidence (perception only),
  point cloud (`deploy/depth_obstacle_node.py` only, not started at boot).
- `config/zed_perception_{front,left,right}.yaml`: 34 / 30 / 30 keys, all
  present in v5.2.2 `common_stereo.yaml` + `zedx.yaml` with matching types.

## 3. `perception_stack.launch.py` with stand-in hardware

Real launch file, colcon-built; `zed_wrapper` and `avros_bringup` replaced by
stubs that start named nodes and report the arguments they received.
ROS domain 78.

**A — first person, full stack** (`camera_start_delay:=3 sensors_settle:=2`):

    domain=78 cameras=['front', 'left', 'right'] already running=[] start sensors=True costmap=True
    [fake sensors] velodyne=true zed_front=false zed_left=false zed_right=false
    zed_front starts in 2 s / zed_left in 5 s / zed_right in 8 s
    zed_front STARTED serial=42569280 override=zed_perception_front.yaml
        publishing=[depth_confidence, depth_map, rgb] pos_tracking=False
    (left 49910017, right 43779087 likewise, 3 s apart)

**B — second person, same domain, same command:** everything reused.

    sensors already running -- reusing
    costmap node already running -- reusing
    cameras=[] already running=['front', 'left', 'right'] start sensors=False costmap=False

**C — third person, domain 79, `cameras:=front,right`:** invisible over ROS,
caught by the process-list check.

    domain=79 cameras=[] already running=['front', 'right']

**Ctrl-C:** launch exited within 0.5 s and all 4 child processes were gone.
(An earlier "launch ignores SIGINT" result was a test artifact: non-interactive
bash starts background jobs with SIGINT ignored. The re-test restored the
default handler first.)

**Errors:** `cameras:=front,back` → "unknown camera(s) back; choose from front,
left, right". No zed_wrapper → "cameras:=front needs zed_wrapper; source its
install, or pass cameras:=''".

## Not covered

Real ZED X hardware and the GMSL timing; whether `depth_stabilization: 0`
depth is clean enough; Humble; any measurement on the Orin.
