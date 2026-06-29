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

# point it at CARLA / real sensor topics
ros2 launch perception_costmap perception.launch.py \
    image_topic:=/carla/ego/rgb_front/image \
    lidar_topic:=/carla/ego/lidar \
    rviz:=true
```

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
PYTHONPATH=.:$PYTHONPATH python3 -m pytest test -q     # 11 offline tests
```

## CARLA smoke test (on the x86 / 5090 box)

1. Start CARLA, spawn the ego with a front RGB camera + lidar.
2. `ros2 launch perception_costmap perception.launch.py image_topic:=<cam> lidar_topic:=<lidar> rviz:=true`
3. In RViz add the `/perception/costmap` OccupancyGrid display. You should see
   the road as free (green-ish), the off-road and any vehicles as lethal.
4. Drive the ego; the costmap should track the road ahead and mark obstacles.
5. Then load `nav2_costmap_params.yaml` into Nav2 and confirm the local
   costmap reflects the same road/obstacles.

## Status

- Geometry, segmentation, obstacle detection, the node, build, and 11 tests:
  done and verified offline + under ROS2.
- Not yet done: tuning the IPM against a real CARLA camera, the learned
  (TwinLiteNet) segmentation backend, and the on-Jetson sensor drivers.
