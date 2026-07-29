#!/usr/bin/env python3
"""Minimal Nav2 bringup for the real car: controller_server, planner_server,
behavior_server, bt_navigator + lifecycle_manager. Mirrors sim_nav2_launch.py.

SAFETY: controller_server's /cmd_vel output is remapped to /nav2/cmd_vel, an
isolated topic actuator_node does NOT subscribe to (it only listens on the
real /cmd_vel). This is deliberate -- prep/validation only, per instruction
not to connect Nav2 to the real motors until explicitly told to. To actually
enable driving later, remap /nav2/cmd_vel -> /cmd_vel (one line, reversible,
visible) -- do not do this without being told to.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
# self-locating: config/ sits next to deploy/ in the package, so this
# resolves correctly from any clone location (no hardcoded home path).
PARAMS = os.path.normpath(os.path.join(_HERE, "..", "config", "real_nav2_params.yaml"))

# Safety default is DISARMED: velocity output goes to /nav2/cmd_vel, which
# actuator_node does not subscribe to, so Nav2 cannot move the vehicle.
# Setting NAV2_ARMED=1 publishes to the real /cmd_vel instead. Keeping this
# an explicit opt-in means "connected to the motors" is never the state you
# end up in by forgetting something.
ARMED = os.environ.get("NAV2_ARMED") == "1"
CMD_VEL_REMAP = [] if ARMED else [("/cmd_vel", "/nav2/cmd_vel")]


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="nav2_controller",
            executable="controller_server",
            name="controller_server",
            output="screen",
            parameters=[PARAMS],
            remappings=CMD_VEL_REMAP,
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            name="planner_server",
            output="screen",
            parameters=[PARAMS],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            name="behavior_server",
            output="screen",
            parameters=[PARAMS],
            remappings=CMD_VEL_REMAP,
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            name="bt_navigator",
            output="screen",
            parameters=[PARAMS],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager",
            output="screen",
            parameters=[PARAMS],
        ),
    ])
