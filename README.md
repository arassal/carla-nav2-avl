# CARLA + ROS2 Navigation Stack
## AVL Mentor Project - Professional Sim-to-Real Pipeline

![Architecture](assets/architecture-diagram.svg)

Professional autonomous navigation research platform developed by the AVL mentor team. This is a production-grade implementation of autonomous driving in simulation, designed for seamless transfer to real-world vehicle deployment.

## 🎯 Project Goal

**Sim-to-Real Autonomous Navigation**: Test and validate autonomous driving algorithms in CARLA simulator using the exact camera layout of our real vehicle, then deploy the same ROS2 stack to the physical car.

![Pipeline](assets/sim-to-real-pipeline.svg)

## 🚀 Key Features

![Features](assets/features-showcase.svg)

- **Lane Following** - Regulated pure pursuit controller with adaptive lookahead
- **Obstacle Avoidance** - Real-time lidar-based 2D costmap and emergency braking
- **Traffic Lights** - Automatic detection and enforcement of traffic light states
- **Real-time Visualization** - Live RViz costmap views and path planning debug
- **Modular Architecture** - ROS2-based design that works in sim and on real hardware
- **20 Hz Control Loop** - Professional-grade deterministic control cycle

## 📋 System Overview

![System](assets/system-overview.svg)

### Hardware Setup
- **3x USB Cameras** (Front, Left, Right @ 1280×720 30Hz, 110° FOV)
- **Lidar** (360° native ROS2 integration)
- **NVIDIA RTX GPU** (5090 recommended, 5070 Ti validated)

### Software Stack
- **CARLA 0.10.0** (Unreal Engine 5) - High-fidelity simulator
- **ROS2 Humble** - Robotics middleware (domain 0, native DDS)
- **Navigation2 (Nav2)** - Production-grade navigation stack
- **Custom Bridge** - CARLA Python API to ROS2 interface

## ⚡ Quick Start

```bash
# Clone the repository
git clone https://github.com/arassal/carla-nav2-avl.git
cd carla-nav2-avl

# Start CARLA simulator (in separate terminal)
cd ~/carla
./CarlaUE4.sh -quality-level=Low

# Build ROS2 workspace
cd carla-nav2-avl/ros2_ws
colcon build

# Run the full stack
source install/setup.bash
../scripts/run_stack.sh
```

## 📊 Project Status

- ✅ **Complete** - Professional ROS2 + Nav2 stack
- ✅ **Original** - Built from scratch by team (no external code)
- ✅ **Tested** - Validated on CARLA Town10HD
- 🔄 **In Integration** - Ready for team development

## 👥 Team

**Leader**: alexander (@arassal)  
**Mentees**: jchy05, AdamCastillo07, adrian (@Ad-Tap)

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for details.

See [CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md) for development workflow.

## 📋 Requirements

| Component | Version |
|-----------|---------|
| **OS** | Ubuntu 22.04 LTS |
| **ROS2** | Humble |
| **CARLA** | 0.10.0 (UE5) |
| **GPU** | NVIDIA RTX (5090/5070 Ti tested) |
| **RAM** | 30GB+ |
| **VRAM** | 12GB+ |
| **CUDA** | 12.x |

## 📁 Repository Structure

```
carla-nav2-avl/
├── ros2_ws/src/              # ROS2 workspace
│   ├── world_setup/          # CARLA bridge
│   ├── controller/           # Nav2 nodes & control
│   ├── sdc_bringup/          # Launch files
│   └── carla_msgs/           # Message definitions
├── scripts/                  # Launch & utility scripts
├── assets/                   # Visuals & diagrams
├── CONTRIBUTORS.md           # Team info
├── CONTRIBUTION_GUIDE.md     # Development guide
└── README.md                 # This file
```

## 🔧 Architecture

**3-Layer Design**:

1. **CARLA Simulator** - Physics, sensors, traffic
2. **CARLA Bridge** - Python API → ROS2 interface
3. **ROS2 + Nav2** - Perception, planning, control (runs on real car too)

## 📖 Documentation

- [Architecture Deep Dive](docs/architecture.md) - System design details
- [Setup Guide](docs/setup.md) - Installation & build instructions
- [Contribution Guide](CONTRIBUTION_GUIDE.md) - How to contribute
- [API Reference](docs/api.md) - ROS2 topics and services

## 📝 License

Original work by AVL mentor team 2026
| Python    | 3.10 (rclpy) / 3.11 (CARLA API) |
| CARLA     | 0.10.0 (Unreal Engine 5.5), built from source |

It has **not** been tested on other hardware, OS versions, GPUs or CARLA
builds yet — treat the steps below as specific to a setup like the above.

## Installing CARLA UE5 with native ROS 2

CARLA's native ROS 2 interface is a UE5-only feature, so CARLA UE5 has to
be built from source against the matching Unreal Engine fork.

1. **Unreal Engine for CARLA** — clone CARLA's UE5 fork
   (`CarlaUnreal/UnrealEngine`, the `ue5-dev-carla` branch) and build it.
   It is large (160+ GB built) and slow. Export its path as
   `CARLA_UNREAL_ENGINE_PATH`.
2. **CARLA** — clone `carla-simulator/carla` (`ue5-dev` branch). Run the
   setup, then build the package target. Configure with a recent CMake
   (the build needs CMake >= 3.28; the distro 3.22 is too old and fails at
   configure). Export the CMake `bin` directory onto `PATH` and
   `CARLA_UNREAL_ENGINE_PATH` **before** invoking CMake — non-interactive
   shells do not source the profile, and the FastDDS reconfigure step
   needs that variable.
3. **Python build deps** — CMake's `FindPython` picks Python 3.11; install
   `build`, `numpy`, `scikit-build-core` and a recent `pip` for that
   interpreter via `uv` (not bare `pip`).
4. **Package build** — only a packaged build yields a runnable standalone
   server (the editor `launch` target opens the GUI; the bare game binary
   has no cooked content). The package lands under
   `Build/Package/.../CarlaUnreal.sh` and is launched with `--ros2`.

Run the server (windowed, native ROS 2 on the RPC port):

```bash
scripts/run_carla_server.sh        # or the packaged CarlaUnreal.sh --ros2
```

### Issues we hit (and fixes)

- **CMake too old / env not sourced** — non-interactive builds don't read
  the shell profile. Export the new CMake `PATH` and
  `CARLA_UNREAL_ENGINE_PATH` explicitly before every CMake call.
- **Cook runs out of memory** — cooking every map OOMs on a 30 GB box.
  Trim the maps-to-cook list in the project's `DefaultGame.ini` to just
  the map(s) you need, and add swap.
- **A non-executable `env` shim on `PATH`** shadowed coreutils `env` and
  broke UE packaging ("Permission denied" on `.../bin/env`). Remove/rename
  any such shim from `PATH`.
- **Native ROS 2 is on DDS domain 0.** If your shell forces a different
  `ROS_DOMAIN_ID`, export `ROS_DOMAIN_ID=0` for every ROS 2 CLI/node that
  must see CARLA topics.
- **Native ROS 2 here is sensor-publish-oriented** — there is no native
  vehicle-control topic or odometry. Driving the ego goes through the
  CARLA Python API; that is why `carla_bridge.py` exists.
- **Sensor naming** — a sensor's `ros_name` becomes its DDS topic. A
  camera with a `ros_name` is routed to DDS and its Python callback never
  fires; spawn the viz camera unnamed (or just use the auto-named native
  camera topic).
- **synchronous_mode** — only the client that ticks the world sees freshly
  spawned actors and valid transforms; settle the world a few ticks before
  reading the spawn pose, and treat the ticking client as authoritative.

## Environment variables / paths

Set these before building/running (non-interactive shells do not read
your profile, so export them explicitly):

```bash
# CARLA UE5 build: point at the Unreal Engine fork you built
export CARLA_UNREAL_ENGINE_PATH=/path/to/CarlaUnreal/UnrealEngine
# the build needs CMake >= 3.28; put it ahead of the distro CMake
export PATH=/path/to/cmake-3.28+/bin:$PATH
# CARLA native ROS 2 publishes on DDS domain 0
export ROS_DOMAIN_ID=0
```

The Python `carla` module path is handled by its wheel/`.pth`; no manual
path is needed for it. The only path that must be exported by hand is
`CARLA_UNREAL_ENGINE_PATH` (for the CARLA build) and the Rerun viewer
directory (below).

## Python environment note

ROS 2 `rclpy` runs on Python 3.10 while the CARLA 0.10.0 wheel is Python
3.11 only. The split is deliberate: the CARLA-API bridge runs on 3.11, the
ROS 2 nodes on 3.10, and they communicate over UDP / shared memory.

## Visualisation: installing Rerun

`rerun-sdk` must be installed for the **Python 3.10** interpreter ROS 2
uses (the Rerun node is a ROS 2 node):

```bash
# preferred:
uv pip install --python /usr/bin/python3 rerun-sdk
# if a system install hits a permission error, install into the user
# site-packages instead:
uv pip install --python /usr/bin/python3 \
  --target "$HOME/.local/lib/python3.10/site-packages" rerun-sdk
```

**Important — the Rerun viewer binary must be on `PATH`.** The wheel
bundles the viewer, but a `--target` install leaves it off `PATH`, and
`rr.init(spawn=True)` then fails with *"Failed to find Rerun Viewer
executable in PATH"*. Add the bundled CLI directory to `PATH`:

```bash
export PATH="$HOME/.local/lib/python3.10/site-packages/rerun_sdk/rerun_cli:$PATH"
```

`scripts/run_rerun.sh` and `scripts/run_stack.sh` already prepend this
directory, so running through the scripts works without extra steps.

The CARLA built-in agents used during development also need `shapely`
and `networkx`; install those the same way for the Python 3.11
interpreter if you use that path.

## Building the ROS 2 workspace

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Running

Start the CARLA server, then bring up the whole stack:

```bash
scripts/run_carla_server.sh          # wait until the RPC port is up
scripts/run_stack.sh                 # bridge + Nav2 + nodes + Rerun
```

`run_stack.sh` kills any stale instances first and starts exactly one of
each component (duplicate ROS 2 nodes and stale viewers caused subtle
bugs, so the script is deliberately strict and timeout-guarded).

Other scripts:

- `scripts/run_nav2.sh` — just the Nav2 side (run after the bridge).
- `scripts/run_rerun.sh` — the Rerun visualisation.
- `scripts/run_viz.sh` — an alternative RViz visualisation.
- `scripts/validate2.sh` — an end-to-end drive + obstacle-brake check.
- `scripts/nav2.yaml` — Nav2 controller / costmap parameters (tune speed,
  lookahead, costmap footprint here).

To verify it is genuinely autonomous (not a scripted path): respawn at a
different point and it drives a different road; spawn an obstacle ahead at
runtime and it brakes; nudge it off the lane and the controller steers it
back; `ros2 topic echo /cmd_vel` shows control recomputed live.

## Notes and limitations

- Localisation uses CARLA's ground-truth pose (no SLAM/EKF). Lane keeping
  uses the HD-map lane centreline (map-based autonomy, as in production AV
  stacks) rather than vision lane detection.
- It is a lane follower: it stops behind a vehicle blocking its lane but
  does not overtake or globally re-plan around it.
- CARLA UE5 is heavy; expect a slow first launch (shader compilation) and
  significant VRAM use.
