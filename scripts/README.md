# scripts/ — legacy CARLA-sim helpers

These are helper scripts from the **CARLA simulation** phase (running the
stack against CARLA on an x86 box, before real-car deployment). They are kept
for reference and are **not** the current entry points.

The live, on-car entry points moved into
`ros2_ws/src/perception_costmap/deploy/` — use those:
- perception + Nav2 bring-up: `deploy/real_nav2_launch.py`
- full on-car stack (tmux): `deploy/full_stack_restart.sh`
- costmap RViz: `deploy/open_costmap.sh`

Treat anything here (`run_carla.sh`, `run_stack.sh`, `nav2.yaml`, `sdc.rviz`,
`validate*.sh`, `spike_topics.sh`, …) as sim-era scratch unless you are
specifically working in CARLA.
