# Costmap tick rate vs CPU, measured on the car

**Date:** 2026-09-17 · **Car:** dinosaur · **Method:** cameras left running
untouched (front 8 Hz, left/right 15 Hz, boot-stack profile); only the costmap
node restarted at each rate with `publish_rate` overridden. CPU read from
`/proc` over a fixed 25 s window (100% = one core).
Script: `~/chris_test/sweep_rate.sh` on the car.

| tick rate | measured | costmap_node CPU | depth/IPM projections |
|---|---|---|---|
| 10 Hz | 9.93 | **67%** | 134 / 136 |
| 8 Hz | 7.82 | 59% | 66 / 72 |
| 6 Hz | 5.92 | 54% | 78 / 94 |
| 5 Hz | 5.02 | **48%** | 83 / 103 |

Halving the tick rate saves **0.19 of a core** and doubles how stale Nav2's
obstacle picture is. Keep 10 Hz; 8 Hz is defensible, below that the trade is
bad.

The frame counters (`inference=submitted/replaced/completed`) are not
comparable across rates here: the node logs them every 100 published ticks, so
the sampling window differs per rate. Use a fixed-duration counter if that
number matters.

## What else was on the machine during this session

**One-off, not how the car normally runs:** a teammate's sensor-fusion stack
and a second RViz were up at the same time for a test. Noted only because it
explains the numbers below and the cameras falling to 14 Hz.

| process | CPU |
|---|---|
| teammate's `voxel_mapper` (sensor-fusion stack) | **275%** |
| its `rviz2` (voxel + lidar displays) | **184%** |
| ZED drivers, 3 cameras, **old profile** | ~140% |
| `costmap_node` | 67-91% |
| `viz_node` / `costmap_rgb` | 23% / 6% |

Load average was 19-20 on 12 cores and the side cameras dropped to 14 Hz
instead of 15.

**With only perception running** (one viewer at most, lean cameras) the same
machine sat at 27.3% CPU mean with the costmap steady at 10 Hz
(`2026-09-16_lean-vs-old-camera-profile.md`). So for a normal run the tick
rate is the smallest lever available: the camera profile is worth ~0.31 core,
the tick rate 0.19 at the cost of half the update rate. Spend the first, keep
the second at 10 Hz.

## Front camera outdoors

12 failures this session, all `CAMERA REBOOTING` / `Connection issue detected`
-- the GMSL link dropping mid-stream, not a failure to open, and not CPU
(tj 62 C, MAXN, no throttling). That signature is a marginal connector, worse
with vibration outdoors. Reseat the front camera's cable at both ends.
`deploy/camera_doctor.sh` reports this case as LINK.
