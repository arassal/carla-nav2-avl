# costmap_node live run on Alexander's modules (0fedca3)

**Date:** 2026-09-14 · **Host:** laptop, Ubuntu 24.04, ROS 2 Jazzy, apt OpenCV 4.6.0, numpy 1.26.4
**Branch:** `feature/christian-costmap-node-modules`, rebuilt on `copy` @ `0fedca3`

## Setup

No sensors. A scratch fixture published a synthetic 640x360 front camera
(grey road on green grass, one dark box on the road) plus `camera_info` at
10 Hz, and `/wheel_odom` at 50 Hz. Default config:

    PYTHONNOUSERSITE=1 python3 -m perception_costmap.costmap_node \
        --ros-args --params-file config/perception_costmap.yaml

(`PYTHONNOUSERSITE=1` only because this laptop has a user-site numpy 2.4 that
breaks ROS's cv_bridge.) A probe watched `/perception/costmap` and
`/diagnostics` for 8 s, called `/perception/reset`, then watched 6 s more.

## Results

How the bugs showed up first (run on this branch's earlier module versions,
same fixture):

| state | lethal / off-road / free / unknown | detector jobs |
|---|---|---|
| before C5 fix | 0 / 0 / 0 / **40000** | — |
| C5 fixed, before C6 fix | 911 / 7221 / 1326 / 25242 | **13** in ~10 s, 84 waits |

With both fixes, on Alexander's modules:

    /perception/costmap   9.4 Hz   861 lethal / 7206 off-road / 1568 free / 25242 unknown
    detectors             OK       55 submitted / 55 completed, 0 errors, 0 depth waits
    camera/front          ERROR    "registered depth stream stale" -- correct per his
                                   health.py: the default config has no depth topic
    reset                 success  301 lethal cells cleared; 57 costmaps in the next 6 s
    SIGINT                         node exited cleanly

## Not covered

Depth/confidence-matched projection, YOLO / cone / TwinLiteNet, multiple live
cameras, Humble, the Jetson.
