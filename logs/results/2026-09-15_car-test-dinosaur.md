# Car test on dinosaur — PRs #1-#4

**Date:** 2026-09-15 · **Car:** dinosaur (Jetson AGX Orin, Ubuntu 22.04, ROS 2 Humble, MAXN)
**Code:** separate checkout `~/chris_test/carla-nav2-avl`, branch `test/all-prs` =
`copy` @ `0fedca3` + PRs #1-#4 merged, built with colcon into its own install.
The boot stack's checkout (`~/carla-nav2-avl`) was not touched. Someone else
was running wheel PID step tests at the same time; Nav2/autodrive was not
started.

## 0. Baseline: the boot stack as found (Alexander's `0fedca3`)

    /perception/costmap                 10.06 Hz
    zed_front / left / right RGB        7.97 / 9.69 / 15.05 Hz
    zed_front depth_registered          7.84 Hz (subscribed by perception_costmap)
    confidence_map, all 3 cameras       advertised   <- ISSUES.md C9 was wrong for the car
    ZED topics per camera               27, incl. point_cloud and odom/pose
    CPU (12 cores)                      ~25-60% per core; GPU (GR3D) 0-92%
    process CPU: costmap_node 103%, rviz2 41%, viz_node 30%, costmap_rgb 10%

## 1. Offline tests on the car

    PYTHONPATH=. python3 -m pytest test -q      104 passed   (system python 3.10)

Conda `(base)` auto-activates for the `dinosaur` user; `conda deactivate` first,
or pytest is missing.

## 2. Costmap node from these PRs, on the real cameras

The boot costmap was stopped (tmux `percept:costmap`) and replaced with this
checkout's node on `perception_dinosaur.yaml`:

    YOLO obstacle detector loaded: /home/dinosaur/models/yolov8n.engine
    cone detector loaded: /home/dinosaur/models/cone_det.engine

**The front ZED had already dropped off GMSL at 12:24:36**, nine minutes before
the swap ("CAMERA STREAM FAILED TO START", watchdog relaunching every ~37 s, 19+
times). The boot costmap stopped publishing at the same moment -- `front` is the
only required camera -- so this was not caused by the test. The node was run
with the left camera required instead:

    ros2 run perception_costmap costmap_node --ros-args --params-file .../perception_dinosaur.yaml \
      -p "required_cameras:=['left']" -p primary_detection_camera:=left

| check | result |
|---|---|
| `/perception/costmap` | **10.0 Hz** (0.07-0.12 s between messages) |
| detector worker | 402 of 408 jobs completed, 5 replaced by newer frames, 0 errors |
| depth pairing | 156 frames matched a depth image, 7 did not |
| side-camera rotation | `yolo=['left', 'right']` |
| `ros2 service call /perception/reset std_srvs/srv/Trigger` | `success=True`; pipeline counters restarted from zero (`sync=149/2`) |
| `deploy/fresh_run.sh --perception` | `CLEAR -- no prior-run state retained`, exit 0 |

The reset reported 0 lethal cells because nothing was in view of the left
camera. TensorRT warned that `yolov8n.engine` was built on a different device.

## 3. Nav2 cloud bridge (`deploy/costmap_to_cloud.py`), fed by the node above

    /perception/costmap_cloud    9.95 Hz, frame base_link
    endpoints per cloud          401: 57-76 marked (road-edge boundary), ~330 clearing at 16 m
    nearest endpoint             1.45 m

**Bug found:** the first version of the C4 guard warned "18% of
/perception/costmap sits at cost 25 ... ROAD-KEEPING IS DISABLED". Cost 25 is
`unknown_cost` (blind spots); `offroad_cost` was a correct 97. Fixed in PR #1 by
reading the node's `offroad_cost` parameter instead. Re-run on the car:

    perception offroad_cost 97 >= obstacle_threshold 97: road edges reach Nav2
    /perception/costmap_cloud    9.83 Hz

## Not tested yet

- The full click-to-drive path (Nav2 + campus_navigator + RViz click): waits
  until the car is free, because the collision monitor publishes `/cmd_vel`.
- The lean launch (PR #4): needs the whole boot stack stopped.
- A reset with obstacles in view (N > 0 lethal cells cleared).
- The front camera path: camera down during the test.

## Hardware notes for the team

- zed_front: GMSL failure from 12:24:36, not recovering by relaunch. Recovery
  is `deploy/clean_camera_restart.sh` (sudo, restarts all three cameras).
- zed_right rebooted 3 times and zed_left once in the preceding hour.
