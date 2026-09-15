"""
launch_guard.py
---------------
Decides what launch/perception_stack.launch.py still needs to start, so two
people bringing the stack up on the same car don't start the same hardware
twice.

Starting a second driver for a ZED X that is already streaming wedges the
GMSL link ("CAMERA STREAM FAILED TO START") and takes the running camera down
with it. Two checks, because either alone misses a case:

  * the ROS graph -- sees nodes on OUR ROS_DOMAIN_ID, from any machine
  * the local process list -- sees ZED drivers on ANY domain, on this machine

Pure Python, ROS-free: the launch file gathers the facts, this decides.
"""

import os
import re

CAMERA_NAMES = ("front", "left", "right")

_CAMERA_ARG = re.compile(r"camera_name:=zed_([A-Za-z0-9_]+)")
_CONTAINER_NS = re.compile(r"__ns:=/zed_([A-Za-z0-9_]+)")


def parse_camera_list(text):
    """'front, left' -> ['front', 'left']. Order kept, duplicates dropped.
    Raises ValueError naming any camera that isn't on the car."""
    names = []
    for raw in str(text).split(","):
        name = raw.strip()
        if name and name not in names:
            names.append(name)
    unknown = [n for n in names if n not in CAMERA_NAMES]
    if unknown:
        raise ValueError("unknown camera(s) %s; choose from %s"
                         % (", ".join(unknown), ", ".join(CAMERA_NAMES)))
    return names


def cameras_in_cmdlines(cmdlines):
    """Camera names with a ZED driver in any of these process command lines.

    Matches both ways a driver appears: the `ros2 launch zed_wrapper
    zed_camera.launch.py camera_name:=zed_front` process, and the component
    container it spawns with `-r __ns:=/zed_front`.
    """
    found = set()
    for cmd in cmdlines:
        if "zed_camera.launch.py" in cmd:
            found.update(_CAMERA_ARG.findall(cmd))
        if "component_container" in cmd:
            found.update(_CONTAINER_NS.findall(cmd))
    return found


def local_cmdlines(proc_root="/proc"):
    """Command lines of every process we can read, NUL separators as spaces."""
    lines = []
    try:
        pids = [p for p in os.listdir(proc_root) if p.isdigit()]
    except OSError:
        return lines
    for pid in pids:
        try:
            with open(os.path.join(proc_root, pid, "cmdline"), "rb") as f:
                raw = f.read()
        except OSError:  # exited, or not ours to read
            continue
        if raw:
            lines.append(raw.replace(b"\0", b" ").decode("utf-8", "replace"))
    return lines


def cameras_on_graph(nodes):
    """Camera names whose driver node is on the ROS graph.
    ``nodes`` is a set of (namespace, name) pairs."""
    return {ns[len("/zed_"):] for ns, name in nodes
            if name == "zed_node" and ns.startswith("/zed_")}


def plan(requested_cameras, running_cameras, start_sensors, start_costmap,
         camera_delay, sensors_settle):
    """What to start and when.

    Returns a dict:
      cameras   [(name, start_after_sec), ...] in requested order
      skipped   camera names already running
      sensors   bool
      costmap   bool

    Cameras start one at a time, ``camera_delay`` seconds apart -- parallel
    ZED X starts wedge the GMSL streams (DEPLOY.md section 6). If this launch
    also starts sensors, the first camera waits ``sensors_settle`` seconds
    for robot_state_publisher, because the ZED driver blocks until the static
    TF exists.
    """
    to_start = [c for c in requested_cameras if c not in running_cameras]
    skipped = [c for c in requested_cameras if c in running_cameras]
    first = float(sensors_settle) if start_sensors else 0.0
    return {
        "cameras": [(name, first + i * float(camera_delay))
                    for i, name in enumerate(to_start)],
        "skipped": skipped,
        "sensors": bool(start_sensors),
        "costmap": bool(start_costmap),
    }
