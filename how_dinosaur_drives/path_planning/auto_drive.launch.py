"""Bring up the autonomy stack for point-and-go autonomous driving.

This launches the layer that sits ON TOP of what percept-stack.service already
starts at boot (sensors, 3x ZED, perception_costmap, odom EKF, web streams):

  - route_server        graph routing over the CPP campus GeoJSON
  - navsat_transform    GPS -> map-frame Cartesian
  - ekf_filter_node_map GPS+IMU+wheel fusion -> map->odom TF
  - costmap_to_cloud    perception drivable-area grid -> PointCloud2 for Nav2
  - Nav2 servers        controller / planner / behaviors / smoother / bt_navigator
  - lifecycle managers  transition all of the above to ACTIVE
  - RViz                campus graph + live pose + Publish Point tool

The persistent campus_navigator node starts with this launch. Once RViz opens,
selecting a point is the complete operator workflow. The node plans over the
campus graph and continuously hands off rolling-costmap-sized goals to Nav2.

NOTE ON cmd_vel WIRING: controller_server publishes /cmd_vel_nav, the velocity
smoother publishes /cmd_vel_collision_in, and Collision Monitor alone publishes
/cmd_vel for actuator_node. This keeps acceleration limiting and the independent
near-field stop/approach filter in series; bypassing either is fail-unsafe.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


# Lives in the sibling perception repo, not this one. Parameterised so a
# different checkout location does not require editing this file.
DEFAULT_CLOUD_BRIDGE = (
    '/home/dinosaur/carla-nav2-avl/ros2_ws/src/perception_costmap/'
    'deploy/costmap_to_cloud.py'
)

NAV2_SERVERS = [
    ('nav2_controller', 'controller_server'),
    ('nav2_smoother', 'smoother_server'),
    ('nav2_planner', 'planner_server'),
    ('nav2_behaviors', 'behavior_server'),
    ('nav2_velocity_smoother', 'velocity_smoother'),
    ('nav2_collision_monitor', 'collision_monitor'),
    ('nav2_bt_navigator', 'bt_navigator'),
]


def generate_launch_description():
    pkg = get_package_share_directory('avros_bringup')
    cfg = os.path.join(pkg, 'config')

    params_file = os.path.join(cfg, 'nav2_params_auto_drive.yaml')
    graph_file = os.path.join(cfg, 'cpp_campus_graph.geojson')
    bt_xml = os.path.join(cfg, 'navigate_igvc_autonav_2026.xml')
    navsat_cfg = os.path.join(cfg, 'navsat_campus.yaml')
    ekf_cfg = os.path.join(cfg, 'ekf.yaml')
    rviz_cfg = os.path.join(pkg, 'rviz', 'auto_drive.rviz')
    cyclonedds = os.path.join(cfg, 'cyclonedds.xml')

    configured = RewrittenYaml(
        source_file=params_file,
        param_rewrites={
            'graph_filepath': graph_file,
            'default_nav_to_pose_bt_xml': bt_xml,
            'default_nav_through_poses_bt_xml': bt_xml,
        },
        convert_types=True,
    )

    cloud_bridge = LaunchConfiguration('cloud_bridge')
    use_rviz = LaunchConfiguration('rviz')

    nav2_nodes = [
        Node(package=p, executable=n, name=n,
             parameters=[configured], output='screen',
             respawn=True, respawn_delay=2.0,
             # see module docstring: keeps velocity_smoother in the path
             remappings=([('cmd_vel', 'cmd_vel_nav')]
                         if n in ('controller_server', 'behavior_server')
                         else [('cmd_vel', 'cmd_vel_nav'),
                               ('cmd_vel_smoothed', 'cmd_vel_collision_in')]
                         if n == 'velocity_smoother'
                         else []))
        for p, n in NAV2_SERVERS
    ]

    return LaunchDescription([
        # The login shell defaults to ROS_DOMAIN_ID=42, while the boot sensor
        # service deliberately runs on domain 0. Pin every process launched
        # here to the sensor domain so the one-command workflow is reliable.
        SetEnvironmentVariable('ROS_DOMAIN_ID', '0'),
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp'),
        SetEnvironmentVariable('CYCLONEDDS_URI', f'file://{cyclonedds}'),

        DeclareLaunchArgument('cloud_bridge', default_value=DEFAULT_CLOUD_BRIDGE,
                              description='Path to costmap_to_cloud.py (perception repo)'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Open RViz with the campus graph + Publish Point tool'),
        DeclareLaunchArgument('graph_range', default_value='300.0',
                              description='Radius (m) of road graph drawn in RViz; '
                                          '0.0 = whole campus (much heavier under software GL)'),
        DeclareLaunchArgument('route_spacing', default_value='12.0',
                              description='Spacing (m) between rolling Nav2 route goals'),
        DeclareLaunchArgument('straight_handoff', default_value='6.0',
                              description='Preemption distance (m) on straight route sections'),
        DeclareLaunchArgument('turn_handoff', default_value='2.5',
                              description='Preemption distance (m) at sharp route turns'),
        DeclareLaunchArgument('goal_check_radius', default_value='0.3',
                              description='Radius (m) checked around each route anchor'),
        DeclareLaunchArgument('blocked_final_wait', default_value='8.0',
                              description='Wait (s) for an occupied destination to clear'),

        # GPS -> map localisation
        Node(package='robot_localization', executable='navsat_transform_node',
             name='navsat_transform', output='screen',
             parameters=[navsat_cfg],
             remappings=[('imu/data', '/imu/data'), ('gps/fix', '/gnss'),
                         ('gps/filtered', '/gps/filtered'),
                         ('odometry/gps', '/odometry/gps'),
                         ('odometry/filtered', '/odometry/global')]),
        Node(package='robot_localization', executable='ekf_node',
             name='ekf_filter_node_map', output='screen',
             parameters=[ekf_cfg],
             remappings=[('odometry/filtered', '/odometry/global')]),

        # Drivable-area vision -> Nav2 ObstacleLayer
        ExecuteProcess(cmd=['/usr/bin/python3', cloud_bridge], output='screen',
                       respawn=True, respawn_delay=2.0),

        # Graph routing
        Node(package='nav2_route', executable='route_server', name='route_server',
             parameters=[configured], output='screen'),

        # Draw the road network so the operator has something to click on.
        # Decimated on purpose -- see route_graph_viz.py for why (llvmpipe).
        # Raise max_range_m (or set 0.0) to see more of campus at more CPU cost.
        Node(package='avros_navigation', executable='route_graph_viz',
             name='route_graph_viz', output='screen',
             parameters=[{'graph_file': graph_file,
                          'publish_nodes': False,
                          'max_range_m': LaunchConfiguration('graph_range')}]),

        *nav2_nodes,

        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_auto_drive', output='screen',
             parameters=[{'autostart': True,
                          'node_names': [n for _, n in NAV2_SERVERS] + ['route_server']}]),

        # Derive the competition safety-light state from the action server.
        # The actuator independently watches this same action status to choose
        # Nav2 cmd_vel over Web UI commands while a goal is active.
        Node(package='avros_navigation', executable='autonomy_monitor',
             name='autonomy_monitor', output='screen', respawn=True,
             respawn_delay=2.0),

        # Persistent no-terminal point-and-go supervisor. It refuses a click if
        # GNSS, localization, vision costmap, lifecycle state, or e-stop is bad.
        Node(package='avros_navigation', executable='campus_navigator',
             name='campus_navigator', output='screen', respawn=True,
             respawn_delay=2.0,
             parameters=[{
                 'route_spacing_m': ParameterValue(
                     LaunchConfiguration('route_spacing'), value_type=float),
                 'straight_handoff_m': ParameterValue(
                     LaunchConfiguration('straight_handoff'), value_type=float),
                 'turn_handoff_m': ParameterValue(
                     LaunchConfiguration('turn_handoff'), value_type=float),
                 'goal_check_radius_m': ParameterValue(
                     LaunchConfiguration('goal_check_radius'), value_type=float),
                 'blocked_final_wait_s': ParameterValue(
                     LaunchConfiguration('blocked_final_wait'), value_type=float),
                 'blocked_goal_cost': 97,
             }]),

        Node(package='rviz2', executable='rviz2', name='rviz2_auto_drive',
             arguments=['-d', rviz_cfg], output='screen',
             condition=IfCondition(use_rviz),
             additional_env={'LIBGL_ALWAYS_SOFTWARE': '1',
                             '__GLX_VENDOR_LIBRARY_NAME': 'mesa',
                             'GALLIUM_DRIVER': 'llvmpipe'}),
    ])
