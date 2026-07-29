# ros2_ws — ROS 2 workspace

**Only one package here is real and maintained:**

| package | status |
|---|---|
| `src/perception_costmap` | **REAL** — the camera+lidar → costmap → Nav2 stack. Start here. |
| `src/collision_guard` | stub / legacy — do not build on |
| `src/controller` | stub / legacy (only `carla_localization.py` is non-trivial) |
| `src/route_planner` | stub / legacy |
| `src/sdc_bringup` | stub / legacy (`launch/sdc.launch.py` is not maintained) |
| `src/sdc_common` | stub / legacy |
| `src/world_setup` | stub / legacy (CARLA world setup) |
| `src/carla_msgs` | message defs for the CARLA-sim path |

See `CLAUDE.md` at the repo root for the authoritative real-vs-stub map.
To build just the real package:

    colcon build --packages-select perception_costmap
