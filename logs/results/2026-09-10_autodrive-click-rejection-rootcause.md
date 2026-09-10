# Root cause: "computer-vision Nav2 bridge is stale" on every destination click

Reported symptom: operator clicks a point in RViz to auto-drive across campus,
nothing happens, message says the Nav2 bridge is not connected.

## The chain

`campus_navigator.py:296-297` refuses the mission when `_cloud_t` is 0 or stale:

```python
(self._cloud_t > 0.0 and now - self._cloud_t < self._stale_after,
 'computer-vision Nav2 bridge is stale'),
```

`_cloud_t` is stamped only by `_on_cloud`, subscribed to
`/perception/costmap_cloud` (`campus_navigator.py:123-124`). That topic has
exactly one publisher: `deploy/costmap_to_cloud.py`, started by
`auto_drive.launch.py:137`.

**That bridge cannot start.** Its line 31 is:

```python
from perception_costmap.costmap_cloud import raycast_costmap
```

`perception_costmap/costmap_cloud.py` did not exist, in any commit on any branch.

```
$ python3 deploy/costmap_to_cloud.py
ModuleNotFoundError: No module named 'perception_costmap.costmap_cloud'
```

So: bridge dies on import → `/perception/costmap_cloud` never published →
`_cloud_t` stays `0.0` → preflight fails → **every click rejected**.

## Why it looked like a connection problem and not a crash

`auto_drive.launch.py:137-138` runs the bridge with `respawn=True,
respawn_delay=2.0`. It crash-looped every two seconds indefinitely. Combined
with the `CHANGE 2026-09-01` note in `campus_navigator.py:142-177` — that
rejection reasons were previously only on an invisible `std_msgs/String` topic
— the operator got a dead click and a message about staleness, with the actual
`ModuleNotFoundError` scrolling past in a launch log nobody was watching.

Worth noting the earlier fix (mirroring status to an RViz text marker) did its
job: it is why the reason was visible at all this time.

## Fix

Wrote `perception_costmap/costmap_cloud.py` implementing `raycast_costmap` to
the contract its caller and the surrounding comments specify: one endpoint per
bearing, nearest occupied cell only (never the occlusion shadow behind it), far
clearing endpoint on bearings that hit nothing. Pure numpy, ROS-free, per
CLAUDE.md's core-module rule. 9 offline tests in `test/test_costmap_cloud.py`.

## Verified end-to-end on the laptop — no sensors, no Jetson

`tools/fake_costmap_publisher.py` (new test fixture) stands in for the
perception node:

```
$ python3 tools/fake_costmap_publisher.py --obstacle-x 5.0 &
$ python3 deploy/costmap_to_cloud.py &
$ ros2 topic hz /perception/costmap_cloud
average rate: 10.001
	min: 0.091s max: 0.110s std dev: 0.00423s window: 22

[costmap_to_cloud]: published 140 clouds, latest 386 marked rays,
                    15 clearing rays (frame=base_link)
```

401 points per cloud = 401 bearings (±100° at 0.5°), one endpoint each, as the
design requires. The 15 clearing rays are the drivable corridor straight ahead;
the 386 marked rays are the off-road boundary either side — expected, because
the fixture uses the car config's `offroad_cost: 97`, which equals
`obstacle_threshold: 97`, so off-road *is* a boundary by design.

**This unblocks the preflight check.** `_cloud_t` will now be stamped.

## What this does NOT fix

The bridge consumes `/perception/costmap`, published by `costmap_node.py` —
still dead from the four missing modules in `ISSUES.md` B1. On the car the
chain is:

    costmap_node (B1, still broken)
        -> /perception/costmap
        -> costmap_to_cloud (B2, fixed here)
        -> /perception/costmap_cloud
        -> campus_navigator preflight  -> the click works

So clicking a destination on the real robot still needs B1 resolved. The
fixture above substitutes for that first stage on a laptop only.
