# CARLA Nav2 - Autonomous Navigation Stack

A complete autonomous navigation system built on CARLA, ROS2, and Navigation2. Designed for simulation validation before real hardware deployment.

## What This Does

Three-camera multi-view perception system generates occupancy grids in real-time. The stack builds costmaps from sensor data, plans collision-free paths, and controls the vehicle to follow them. Everything runs through standard ROS2 topics so you can swap the simulator for real hardware without changing the core logic.

## The Setup

**Cameras**: Front, left, right views. Each captures 1280×720 at 30 Hz. This gives you coverage of obstacles in the immediate vicinity and lane boundaries.

**Lidar**: 64-ray lidar gives you a proper 3D view of the environment. The costmap layer fuses this with camera data to build a 2D occupancy grid. Obstacle inflation handles safety margins automatically.

**Costmap**: 100×100 meter grid centered on the vehicle. Free space is 0, obstacles are 255, unknown is 128. The planner uses this to find paths that avoid collisions.

**Planning**: Standard A* on the costmap. Produces a path as a sequence of waypoints. The controller follows it using pure pursuit steering with adaptive lookahead.

**Control Loop**: Vehicle command is a `Twist` message (linear velocity, angular velocity). This gets converted to steering angle and throttle for the simulator, or CAN commands for real hardware.

## Why This Matters

Most hobby autonomous driving projects build closed-loop simulators with custom pathfinding and hand-tuned steering gains. This is production-grade: you're using the same stack that real autonomous vehicles use. Navigation2 is maintained by a large community. Your control gains are tunable parameters, not magic numbers in Python. When you move to hardware, you're not rewriting everything.

## Architecture

Four layers that cleanly separate concerns:

1. **Hardware**: GPU, cameras, lidar, compute. CARLA in simulation, real hardware later.
2. **ROS2**: DDS middleware. Sensors publish messages. Nodes communicate through topics. No direct function calls.
3. **Navigation2**: Costmap generation, path planning, controller. All pluggable components.
4. **Application**: CARLA bridge that owns the simulation tick, converts Twist commands to vehicle control.

The key insight: layers 2-4 don't change when you swap layer 1 from CARLA to real hardware.

## Tech Stack

- **CARLA 0.10.0** (UE5): High-fidelity simulator with native ROS2 sensor publishing
- **ROS2 Humble**: Mature, stable middleware
- **Navigation2**: Industry standard for mobile robot navigation
- **Python 3.10/3.11**: Bridge and custom nodes
- **Ubuntu 22.04**: Stable OS, good ROS2 support

## Quick Start

```bash
git clone https://github.com/arassal/carla-nav2-avl.git
cd carla-nav2-avl

# Start CARLA
cd ~/carla && ./CarlaUE4.sh -quality-level=Low

# Build and run the stack
cd carla-nav2-avl/ros2_ws
colcon build
source install/setup.bash
../scripts/run_stack.sh
```

## Requirements

- Ubuntu 22.04
- ROS2 Humble
- CARLA 0.10.0 (built from source)
- NVIDIA GPU (RTX 5090 tested, RTX 5070 Ti works)
- 32GB RAM, 12GB VRAM minimum
- CUDA 12.x

## What Works

- Lane following on any CARLA town using the OpenDRIVE map
- Obstacle detection from lidar, stops the vehicle if something's in the way
- Traffic light enforcement (reads light state, stops on red)
- Multi-camera perception without any custom vision code
- Costmap generation updates in real-time as the vehicle moves
- Full ROS2 integration (everything on standard topics)

## Tested Scenarios

- All CARLA towns (Town01-Town10)
- Different weather (rain, fog, night)
- Heavy traffic, pedestrians crossing
- Edge cases like occlusion, parked vehicles

## Repository

- Main code: `ros2_ws/src/`
- CARLA bridge: `world_setup/`
- Nav2 configuration: `scripts/nav2.yaml`
- Custom nodes: `controller/`

## Team

- **alexander** (@arassal) — CARLA integration, architecture
- **jchy05** — Nav2 configuration and tuning
- **AdamCastillo07** — Visualization and debugging
- **adrian** (@Ad-Tap) — Control system refinement

See `CONTRIBUTORS.md` and `CONTRIBUTION_GUIDE.md` for details.

## Next Steps

Currently validated in simulation. Real hardware integration is in progress. The architecture supports it directly: swap the CARLA bridge for a real vehicle interface, keep everything else.

---

Code: https://github.com/arassal/carla-nav2-avl

Documentation: See `CONTRIBUTION_GUIDE.md` for development workflow.
