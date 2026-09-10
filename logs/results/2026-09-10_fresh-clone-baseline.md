# Fresh-clone baseline — 2026-09-10

Host: chris laptop, Ubuntu, ROS 2 **Jazzy** installed (project targets Humble),
no CARLA running, no sensors attached. numpy 2.4.6, pytest 7.4.4, OpenCV via pip.
Branch: `copy` @ `c102aad`.

## Offline suites — both green

| suite | command | result |
|---|---|---|
| `perception_costmap` | `PYTHONPATH=. python3 -m pytest test -q` | **59 passed**, 0.25 s |
| `driving_seg` | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest test -q` | **7 passed**, 0.08 s |

Docs claim 39 (six places) and 78 (once) for the first suite — see `ISSUES.md` D1.
No ROS needed for either, as `CLAUDE.md` promises.

Raw: `logs/tests/2026-09-10_pytest-*_chris-linux.log`

## The node itself does not import

```
$ source /opt/ros/jazzy/setup.bash
$ cd ros2_ws/src/perception_costmap && PYTHONPATH=.:$PYTHONPATH \
    python3 -c "import perception_costmap.costmap_node"
ModuleNotFoundError: No module named 'perception_costmap.detection_schedule'
```

Four modules referenced by `costmap_node.py` were never committed — `ISSUES.md` B1.

**The headline: 59 green tests and a node that cannot start are both true at
once.** The suite covers only the ROS-free helpers; nothing imports
`costmap_node`. Test count is not a proxy for "the stack runs."

Raw: `logs/tests/2026-09-10_costmap-node-import-failure_chris-linux.log`

## Not attempted

`colcon build` (wrong ROS distro here), CARLA smoke test, Jetson/TensorRT,
anything needing real sensors.
