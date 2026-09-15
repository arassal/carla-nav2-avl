# C3 / C4 — the two costmap↔Nav2 coupling bugs, fixed and measured

Both are the same root cause: a constant in the perception config has to agree
with a constant in the Nav2 config, across two repos, with nothing checking.

## C4 — road-keeping was silently disabled, and here is what it cost

`costmap_to_cloud.py` forwards a cell to Nav2 only when its cost reaches
`obstacle_threshold` (97). Obstacles are always LETHAL (100) so they always get
through. Off-road is `offroad_cost` — and `perception_costmap.yaml`, the
default config `perception.launch.py` loads, never set it, so it fell back to
**65**. Below the threshold. Never forwarded.

Measured with `tools/fake_costmap_publisher.py`, identical scene (a 4 m-wide
drivable corridor, one obstacle 5 m ahead), only `offroad_cost` changed:

| `offroad_cost` | marked rays | clearing rays | what Nav2 sees |
|---|---|---|---|
| 65 (old default) | **17** | 384 | the obstacle only — road edges invisible |
| 97 (fixed)       | **386** | 15 | obstacle **and** both road edges |

17 of 401 bearings versus 386. The robot would have avoided the obstacle and
driven straight off the road, and nothing in any log would have said why.

`real_nav2_params.yaml:89-93` warned about exactly this scenario in a comment
while the shipped default did it.

### Fix, two parts

1. `perception_costmap.yaml` now sets `offroad_cost: 97`, with a comment
   explaining the coupling and why `occupancy.py`'s 65 default (a
   RViz-legibility choice) is wrong for a grid that feeds a planner.
2. `costmap_to_cloud.py` gained `_warn_if_offroad_below_threshold()` — a
   runtime check, because part 1 fixes today's drift and part 2 catches every
   future one. Off-road is one constant painted over a large area, so a
   mismatch shows up as a single sub-threshold value covering a big fraction of
   the grid. Runs at frame 50 (~5 s), then every 600.

Verified both directions:

```
offroad_cost 65 -> [WARN] 80% of /perception/costmap sits at cost 65, below this
  bridge's obstacle_threshold of 97. ... ROAD-KEEPING IS DISABLED -- Nav2 will
  avoid obstacles but not the road edge. Set offroad_cost >= 97 ...

offroad_cost 97 -> 0 warnings
```

No false positive on the correct config, which matters more than the warning
firing — a check that cries wolf gets ignored.

## C3 — phantom obstacle arc at 16 m

The bridge emits a *clearing* endpoint at exactly `raytrace_range_m` = 16.0 m
on every bearing that hits nothing. It is not an obstacle; it exists so Nav2
raytraces that bearing free. Nav2's `obstacle_max_range` must therefore sit
below 16 m or those endpoints get **marked**.

`nav2_params_auto_drive.yaml:396` had this right (`15.5`) and said why.
`real_nav2_params.yaml` did not:

| costmap | before | after |
|---|---|---|
| local  | `obstacle_max_range: 20.0`, `raytrace: 25.0` | `15.5` / `16.0` |
| global | `obstacle_max_range: 30.0`, `raytrace: 35.0` | `15.5` / `16.0` |

Under `real_nav2_launch.py`, every clear bearing was planting a fake obstacle
at 16 m — an arc across the whole ±100° fan. That is the same "wall the planner
cannot get past" failure the ray-cast design exists to avoid, reintroduced one
layer downstream.

Real hits are emitted at ≤ 15 m, so 15.5 marks every genuine obstacle and no
clearing endpoint. Ranges past 16 m bought nothing regardless: the cloud never
contains a point beyond 16 m.

**Not verified live** — needs Nav2 installed and a running ObstacleLayer. The
range arithmetic is checkable by hand; a `ros2 topic echo` of the marked
costmap would confirm it, and `nav2_params_auto_drive.yaml` is the precedent.

## Status

68 offline tests pass. C3 is config-only. C4's runtime check is exercised in
both directions above.

Raw: `logs/tests/2026-09-10_bridge-offroad{65-warns,97-clean}_chris-linux.log`
