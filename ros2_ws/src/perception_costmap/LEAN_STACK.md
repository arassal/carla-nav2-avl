# Lean perception stack — quick reference

One launch that starts **only what perception needs**: sensors, the cameras you
ask for with lean ZED settings, and the costmap node. Measured on dinosaur
2026-09-16: camera drivers **80% → 49% of a core (−39%)**, RAM **−371 MB**,
costmap steady at 10 Hz.

## Run it

```bash
conda deactivate                     # the dinosaur account auto-activates conda
source /opt/ros/humble/setup.bash
source ~/IGVC/install/setup.bash
source <this workspace>/install/setup.bash
export ROS_DOMAIN_ID=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp AVL_MODELS_DIR=/home/dinosaur/models
export CYCLONEDDS_URI=file://$HOME/IGVC/install/avros_bringup/share/avros_bringup/config/cyclonedds.xml

# stop the boot stack first -- it owns the cameras
sudo systemctl stop percept-stack
tmux -L percept kill-session -t percept 2>/dev/null; pkill -x rviz2; sleep 5

ros2 launch perception_costmap perception_stack.launch.py            # 3 cameras + costmap
ros2 launch perception_costmap perception_stack.launch.py rviz:=true # + colorizer + RViz
```

Cameras start staggered (15 s / 55 s / 95 s) — parallel starts wedge GMSL.
Restore the normal stack with `sudo systemctl start percept-stack`.

### Arguments worth knowing

| arg | default | what it does |
|---|---|---|
| `cameras` | `front,left,right` | which cameras; `none` for sensors + costmap only |
| `config` | `perception_dinosaur.yaml` | perception params |
| `rviz` | `false` | colorized costmap view, started **after** the cameras |
| `viz` | `auto` | `costmap_rgb_node` only; `auto` = on when `rviz:=true` |
| `lidar` | `true` | Velodyne. Without the EKF it logs `"odom" does not exist`; perception doesn't use it |
| `sensors` | `true` | URDF/TF, Velodyne, Xsens |
| `costmap_rate` | *(config)* | override `publish_rate` (Hz). Keep >= the fastest camera's rate or frames go unprocessed |
| `reuse_running` | `true` | skip anything already running instead of starting a second copy |

`--show-args` lists the rest (serials, delays, `ros_domain_id`).

## What we changed

- **`config/zed_perception_{front,left,right}.yaml`** — cameras publish only RGB,
  depth and confidence. No point cloud, no positional tracking, no IMU;
  processing capped at 8 fps. Those were tuned for the old kiwicampus Nav2
  layer, which nothing reads any more.
- **`launch/perception_stack.launch.py`** — the launch above. Reuses anything
  already running (ROS graph + local processes), so two people can't start the
  same camera twice.
- **`deploy/perception_lean.rviz`** — costmap as flat squares in `base_link`,
  no RobotModel, no camera panels.
- **`/perception/reset` + `deploy/fresh_run.sh`** — clear a run's accumulated
  obstacles without restarting the stack.
- **Fixes:** C1 grid-edge truncation, C3 Nav2 16 m clearing range, C4
  road-keeping (bridge now reads the node's `offroad_cost` instead of guessing),
  C5 homography scale, C6 depth wait on cameras without depth, C7 detector
  results surviving a reset. Details in `ISSUES.md`.

## Cameras won't open? Ask the doctor

```bash
deploy/camera_doctor.sh          # name the failure and the fix
deploy/camera_doctor.sh --stop   # stop every camera client gracefully first
```

It reads nvargus-daemon and the wrapper logs and says which of four cases you
have: **HELD** (another process owns the camera), **LINK** (GMSL dropping
mid-stream -> reseat the cable), **GL** (a camera opened while something held
the display's GL context -> start RViz after the cameras), or **WEDGED**
(reboot; do *not* restart the camera daemons).

Always stop cameras with `--stop` (SIGINT, wait) rather than
`tmux kill-session` or `pkill -9`: a hard kill leaves Argus holding streams
the next open then trips over.

## Gotchas that cost us time

| Symptom | Cause | Fix |
|---|---|---|
| `CAMERA STREAM FAILED TO START`, `AlreadyAllocated ... in use` | another program holds that ZED | stop it; one client per camera |
| `Cannot create camera provider` / `NOT DETECTED` | camera daemon not answering | reboot; **don't** restart nvargus/zed_x_daemon, that made it worse |
| `(Argus) BadParameter ... createBuffer`, container exit `-11` | a camera opened while RViz held the display's GL context | start RViz **after** the cameras |
| rviz2 "failed to create drawable" then segfault | config with a RobotModel (`zedx.stl` isn't installed), or a second rviz2 on the same display | use `perception_lean.rviz`; `pkill -x rviz2` first |
| RViz opens but the costmap panel is empty | that view draws `/viz/costmap_rgb`; `costmap_rgb_node` isn't running, or the display is unticked | `viz:=true`, and check the display's checkbox |
| `ros2` finds nothing / pytest missing | conda `(base)` is active | `conda deactivate` |

Don't point RViz's Map display at `/perception/costmap`: `unknown_cost` is 25,
so blind cells render as ordinary low cost — blind spots look drivable.
`costmap_rgb_node` uses `/perception/known` to tell them apart.

## Still open

- `depth_stabilization: 0` in the lean profile isn't judged for depth quality.
- Only ~29% of obstacle projections use depth; the rest fall back to IPM
  (`depth_outlier` ~69k per 10 s). Worth investigating for accuracy.
- Costmap node CPU (`ISSUES.md` P2): segmentation re-runs on unchanged frames.
- C8: left/right ZED serials disagree between `sensors.launch.py` and the boot
  scripts. Cover one camera and see which topic goes dark.
