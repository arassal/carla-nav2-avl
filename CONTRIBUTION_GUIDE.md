# Contribution Guide

## What this project actually is
The perception + Nav2 obstacle-avoidance stack for our **IGVC** ground robot
(skid-steer, grass course, cones + painted white lines). We prototype in CARLA
and deploy the identical ROS2 stack to the car ("dinosaur", Jetson AGX Orin,
ROS2 Humble). See `README.md` for orientation and the reading order.

> Earlier versions of this guide described a CARLA-0.10 lane-following /
> traffic-light project. That was aspirational and never built — ignore any
> reference to lane following, traffic lights, or multi-vehicle coordination.

## What's real vs. stub
Only `ros2_ws/src/perception_costmap/` (and `driving_seg/`) are real and
tested. The other `ros2_ws/src/*` packages (`collision_guard`, `controller`,
`route_planner`, `sdc_bringup`, `sdc_common`, `world_setup`) are stubs — do
not build on them without checking `CLAUDE.md` and `ros2_ws/README.md` first.

## Branches
- `main` — integration of validated work (what you get on a default clone).
- `feature/<name>` — per-contributor feature branches.
Open a PR from your feature branch into `main`; keep the branch focused.

## Before you open a PR
- Run the offline tests: `cd ros2_ws/src/perception_costmap && PYTHONPATH=. python3 -m pytest test -q` (39 should pass).
- Keep changes portable: no hardcoded `/home/<user>` paths on the run path
  (nodes, launch files, configs). The `deploy/*.sh` scripts are the one
  documented exception (as-run, host-specific).
- Match the surrounding code's style and comment density.

## Commit / PR conventions
- Small, reviewable commits with a clear "why" in the message.
- Reference the config/file you touched and any hardware assumption you rely on.
