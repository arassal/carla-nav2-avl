#!/usr/bin/env python3
"""Nav2 bringup for the real car: the obstacle-cloud bridge + the five Nav2
servers (controller, planner, behavior, bt_navigator, lifecycle_manager).

The bridge (costmap_to_cloud.py) is started HERE on purpose: Nav2's
ObstacleLayer takes its only observation source from /perception/costmap_cloud
(see real_nav2_params.yaml), which nothing else publishes. Without it Nav2
comes up with an empty costmap and plans through everything -- so the launch
that starts Nav2 must also start the bridge.

SAFETY: controller_server + behavior_server /cmd_vel output is remapped to
/nav2/cmd_vel, an isolated topic actuator_node does NOT subscribe to (it only
listens on the real /cmd_vel). This is the default (disarmed). Set NAV2_ARMED=1
to publish to the real /cmd_vel and let Nav2 drive the motors -- an explicit,
reversible opt-in. Do not arm without a human on the E-stop.
"""
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

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
        # Perception -> Nav2 bridge: raycasts /perception/costmap to first
        # visible surfaces and publishes /perception/costmap_cloud, the
        # ObstacleLayer's observation source. MUST run for Nav2 to see anything.
        ExecuteProcess(
            cmd=["python3", os.path.join(_HERE, "costmap_to_cloud.py")],
            output="screen",
        ),
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
