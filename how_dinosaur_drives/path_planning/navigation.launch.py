"""Launch full autonomous navigation stack.

Launches:
  - Everything from localization.launch.py (sensors + EKF + navsat)
  - actuator_node (cmd_vel -> Teensy UDP)
  - Nav2 servers (controller, smoother, planner, route_server, behavior,
    velocity_smoother, bt_navigator)
  - Single lifecycle manager (transitions servers in order)
  - foxglove_bridge (WebSocket server for remote visualization)

Nav2 servers are launched directly (not via nav2_bringup) so that
route_server can be included in the lifecycle manager's ordered node list.
The lifecycle manager transitions nodes sequentially, guaranteeing
route_server is active before bt_navigator loads the BT XML.

Cross-distro: ROS_DISTRO selects the correct nav2 params and BT XML.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg_dir = get_package_share_directory('avros_bringup')
    actuator_config = os.path.join(pkg_dir, 'config', 'actuator_params.yaml')
    graph_file = os.path.join(pkg_dir, 'config', 'cpp_campus_graph.geojson')
    cyclonedds_file = os.path.join(pkg_dir, 'config', 'cyclonedds.xml')
    waypoints_file = os.path.join(pkg_dir, 'config', 'waypoints.yaml')

    # Select distro-specific config (params + default BT XML). These are the
    # DEFAULTS for the nav2_params / bt_xml launch args below; both are FILENAMES
    # relative to config/, so either can be overridden at launch without editing
    # this file.
    ros_distro = os.environ.get('ROS_DISTRO', 'humble')
    if ros_distro == 'humble':
        # 2026-06-01: DEFAULT is now the IGVC AutoNav COMPETITION pair (the freeze
        # fix — inflation gradient restored to 0.85/0.65, prompt recovery, 45 s
        # watchdog / 4 retries). The competition params + BT are a MATCHED PAIR and
        # default together. To fall back to the field-test pair (no DQ clock — wider
        # 120 s Timeout, thin 0.4 inflation), launch with BOTH:
        #   nav2_params:=nav2_params_humble.yaml bt_xml:=navigate_igvc_autonav_humble.xml
        # See docs/igvc_autonav_tuning_2026_06_01.md.
        default_nav2_params = 'nav2_params_igvc_autonav.yaml'
        default_bt = 'navigate_igvc_autonav_2026.xml'
    else:
        default_nav2_params = 'nav2_params.yaml'
        default_bt = 'navigate_route_graph.xml'

    use_sim_time = LaunchConfiguration('use_sim_time')
    # bt_xml / nav2_params are FILENAMES (relative to config/), not full paths.
    # Lets you swap configs without editing this file. The IGVC competition pair
    # MUST be selected together:
    #   ros2 launch avros_bringup navigation.launch.py \
    #       nav2_params:=nav2_params_igvc_autonav.yaml \
    #       bt_xml:=navigate_igvc_autonav_2026.xml
    bt_xml_filename = LaunchConfiguration('bt_xml')
    nav2_params_filename = LaunchConfiguration('nav2_params')

    # Rewrite nav2 params with resolved paths and use_sim_time. source_file is a
    # substitution list ([pkg]/config/<nav2_params>) so the params file is
    # selectable at launch time (RewrittenYaml normalizes the list internally).
    configured_params = RewrittenYaml(
        source_file=[pkg_dir, '/config/', nav2_params_filename],
        param_rewrites={
            'use_sim_time': use_sim_time,
            'default_nav_to_pose_bt_xml': [pkg_dir, '/config/', bt_xml_filename],
            'default_nav_through_poses_bt_xml': [pkg_dir, '/config/', bt_xml_filename],
            'graph_filepath': graph_file,
        },
        convert_types=True,
    )

    # Nav2 servers — lifecycle manager transitions these in order,
    # so route_server activates before bt_navigator validates the BT XML
    nav2_servers = [
        ('nav2_controller', 'controller_server'),
        ('nav2_smoother', 'smoother_server'),
        ('nav2_planner', 'planner_server'),
        ('nav2_route', 'route_server'),
        ('nav2_behaviors', 'behavior_server'),
        ('nav2_velocity_smoother', 'velocity_smoother'),
        ('nav2_bt_navigator', 'bt_navigator'),
    ]

    lifecycle_nodes = [name for _, name in nav2_servers]

    nav2_nodes = [
        Node(
            package=package,
            executable=name,
            name=name,
            parameters=[configured_params],
            output='screen',
            respawn=True,
            respawn_delay=2.0,
        )
        for package, name in nav2_servers
    ]

    return LaunchDescription([
        # Force CycloneDDS for every node spawned from this launch (and any
        # child includes). Without this, directly-instantiated Nav2 servers
        # come up on the shell default — usually FastDDS — which doesn't
        # interop cleanly with the CycloneDDS sensor stack and corrupts
        # action goal payloads (poses arrive zeroed). See CLAUDE.md
        # known-issues.
        SetEnvironmentVariable(
            name='RMW_IMPLEMENTATION',
            value='rmw_cyclonedds_cpp'
        ),
        SetEnvironmentVariable(
            name='CYCLONEDDS_URI',
            value='file://' + cyclonedds_file
        ),

        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation clock'
        ),

        DeclareLaunchArgument(
            'bt_xml', default_value=default_bt,
            description='Behavior tree XML filename (in config/ dir). Default '
                        '(humble) is now navigate_igvc_autonav_2026.xml (IGVC '
                        'COMPETITION BT). Pass navigate_igvc_autonav_humble.xml '
                        'for the field-test BT (use WITH '
                        'nav2_params:=nav2_params_humble.yaml — matched pair).'
        ),

        DeclareLaunchArgument(
            'nav2_params', default_value=default_nav2_params,
            description='Nav2 params YAML filename (in config/ dir). Default '
                        '(humble) is now nav2_params_igvc_autonav.yaml (IGVC '
                        'COMPETITION config). Pass nav2_params_humble.yaml for '
                        'the field-test config (use WITH '
                        'bt_xml:=navigate_igvc_autonav_humble.xml — matched pair).'
        ),

        DeclareLaunchArgument(
            'enable_ntrip', default_value='true',
            description='Enable NTRIP client for RTK corrections (public '
                        'MDOT CORS caster). Default true: IGVC 2026 §I.2 '
                        'forbids running your OWN base station, but a public '
                        'NTRIP caster is permitted (user-confirmed 2026-05-30 '
                        '— get the judge ruling in writing). Set false for '
                        'unaided GPS.'
        ),

        DeclareLaunchArgument(
            'enable_velodyne', default_value='true',
            description='Enable Velodyne VLP-16 LiDAR'
        ),

        DeclareLaunchArgument(
            'enable_zed_front', default_value='false',
            description='Enable front ZED X camera'
        ),

        DeclareLaunchArgument(
            'enable_zed_left', default_value='false',
            description='Enable left ZED X camera'
        ),

        DeclareLaunchArgument(
            'enable_zed_right', default_value='false',
            description='Enable right ZED X camera'
        ),

        DeclareLaunchArgument(
            'enable_perception', default_value='false',
            description='Enable avros_perception (requires at least one enable_zed_* camera)'
        ),

        DeclareLaunchArgument(
            'perception_cameras', default_value='front',
            description='Comma-separated camera list for perception_node, e.g. "front,left,right"'
        ),

        DeclareLaunchArgument(
            'enable_mission_manager', default_value='true',
            description='Enable mission_manager waypoint orchestrator '
                        '(drives the BT through waypoints.yaml — robot will move). '
                        'DEFAULT ON (2026-06-01): a bare navigation.launch.py '
                        'AUTO-STARTS the waypoint mission ~20 s after launch. '
                        'Disable for bench/no-drive testing: enable_mission_manager:=false'
        ),

        # Localization (sensors + EKF + navsat)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_dir, 'launch', 'localization.launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'enable_ntrip': LaunchConfiguration('enable_ntrip'),
                'enable_velodyne': LaunchConfiguration('enable_velodyne'),
                'enable_zed_front': LaunchConfiguration('enable_zed_front'),
                'enable_zed_left': LaunchConfiguration('enable_zed_left'),
                'enable_zed_right': LaunchConfiguration('enable_zed_right'),
            }.items(),
        ),

        # avros_perception — semantic segmentation feed for Nav2 (kiwicampus layer)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('avros_perception'),
                '/launch/perception.launch.py',
            ]),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'cameras': LaunchConfiguration('perception_cameras'),
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_perception')),
        ),

        # Actuator bridge
        Node(
            package='avros_control',
            executable='actuator_node',
            name='actuator_node',
            parameters=[
                actuator_config,
                {'use_sim_time': use_sim_time},
            ],
            output='screen',
        ),

        # Nav2 servers
        *nav2_nodes,

        # Lifecycle manager — transitions nodes in list order
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            parameters=[{
                'autostart': True,
                'node_names': lifecycle_nodes,
            }],
            output='screen',
        ),

        # Foxglove bridge (WebSocket on port 8765 for remote visualization)
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        # autonomy_monitor — ALWAYS ON. Drives the IGVC §I.2 safety light by
        # watching the NavigateToPose action-server status and publishing the
        # latched /autonomous_mode flag (FLASH while a goal is active, SOLID
        # otherwise). Source-agnostic: covers mission_manager, RViz "Nav2 Goal",
        # and manual CLI goals alike — so the light flashes in autonomous mode
        # no matter how the goal was sent. Pure observer (no /cmd_vel, no goals),
        # so it's safe to run unconditionally; it never moves the robot.
        Node(
            package='avros_navigation',
            executable='autonomy_monitor',
            name='autonomy_monitor',
            parameters=[{
                'use_sim_time': use_sim_time,
            }],
            output='screen',
        ),

        # mission_manager — DEFAULT ON (2026-06-01). A bare `ros2 launch ...`
        # AUTO-STARTS the waypoint mission ~20 s after launch — THE ROBOT WILL
        # DRIVE through waypoints.yaml. Disable for bench/no-drive testing with
        # enable_mission_manager:=false. The orchestrator owns the waypoint
        # cursor and feeds NavigateToPose goals to the BT (skip-on-failure: an
        # ABORTED/CANCELED waypoint is logged and skipped, not retried).
        #
        # Wrapped in a 20 s TimerAction. Two concurrent races otherwise fire:
        #   (1) navsat_transform advertises /fromLL before its datum is fully
        #       initialized — calling fromLL too early returns (0,0,0) for
        #       every input. ~5-15 s for GPS first-fix + datum lock.
        #   (2) bt_navigator advertises /navigate_to_pose during its inactive
        #       lifecycle transition; goals sent during that window are
        #       REJECTED. Lifecycle manager finishes activation ~10-15 s
        #       after launch.
        # 20 s comfortably clears both. If field testing shows nav2 takes
        # longer to come up, bump this; if shorter, lower it. Single point
        # of tuning.
        TimerAction(
            period=20.0,
            actions=[
                Node(
                    package='avros_navigation',
                    executable='mission_manager',
                    name='mission_manager',
                    parameters=[{
                        'waypoints_file': waypoints_file,
                        'use_sim_time': use_sim_time,
                    }],
                    output='screen',
                    condition=IfCondition(LaunchConfiguration('enable_mission_manager')),
                ),
            ],
        ),
    ])
