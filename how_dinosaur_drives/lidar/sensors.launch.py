"""Launch sensor drivers and robot_state_publisher for the AV2 platform.

Launches:
  - robot_state_publisher (URDF -> static TF)
  - velodyne VLP-16 driver + pointcloud transform
  - xsens MTi-680G IMU/GNSS
  - NTRIP client for RTK corrections (optional)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_dir = get_package_share_directory('avros_bringup')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'avros.urdf.xacro')
    cyclonedds_file = os.path.join(pkg_dir, 'config', 'cyclonedds.xml')
    velodyne_config = os.path.join(pkg_dir, 'config', 'velodyne.yaml')
    zed_front_config = os.path.join(pkg_dir, 'config', 'zed_front.yaml')
    zed_left_config = os.path.join(pkg_dir, 'config', 'zed_left.yaml')
    zed_right_config = os.path.join(pkg_dir, 'config', 'zed_right.yaml')
    xsens_config = os.path.join(pkg_dir, 'config', 'xsens.yaml')
    ntrip_config = os.path.join(pkg_dir, 'config', 'ntrip_params.yaml')

    return LaunchDescription([
        # CycloneDDS shared memory
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
            'enable_ntrip', default_value='false',
            description='Enable NTRIP only after configuring a correction '
                        'source near the operating site. The former Michigan '
                        'caster must not be used at Cal Poly Pomona.'
        ),

        DeclareLaunchArgument(
            'enable_velodyne', default_value='true',
            description='Enable Velodyne VLP-16 LiDAR'
        ),

        DeclareLaunchArgument(
            'enable_zed_front', default_value='false',
            description='Enable front ZED X camera (requires ZED SDK + ZED Link Quad)'
        ),

        DeclareLaunchArgument(
            'enable_zed_left', default_value='false',
            description='Enable left ZED X camera (requires ZED SDK + ZED Link Quad)'
        ),

        DeclareLaunchArgument(
            'enable_zed_right', default_value='false',
            description='Enable right ZED X camera (requires ZED SDK + ZED Link Quad)'
        ),

        # robot_state_publisher: URDF -> static TF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{
                'robot_description': ParameterValue(
                    Command(['xacro ', urdf_file]), value_type=str
                ),
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }],
            output='screen',
        ),

        # Velodyne VLP-16 driver
        Node(
            package='velodyne_driver',
            executable='velodyne_driver_node',
            name='velodyne_driver_node',
            parameters=[velodyne_config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_velodyne')),
        ),

        # Velodyne raw packets -> PointCloud2
        Node(
            package='velodyne_pointcloud',
            executable='velodyne_transform_node',
            name='velodyne_transform_node',
            parameters=[velodyne_config],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_velodyne')),
        ),

        # ZED X Front (GMSL via ZED Link Quad). Launched via the wrapper's own
        # launch file — zed_wrapper is a metapackage; the camera is a composable
        # component (stereolabs::ZedCamera) loaded by zed_camera.launch.py.
        #
        # Canonical args per zed-ros2-examples: pass only camera_model +
        # camera_name; wrapper sets namespace=camera_name, node_name=zed_node,
        # and our URDF provides the frame chain via zed_macro.urdf.xacro.
        # Passing `namespace` + `node_name` explicitly triggers a collision
        # (wrapper overwrites node_name with camera_name → /zed_front/zed_front/...).
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('zed_wrapper'),
                '/launch/zed_camera.launch.py',
            ]),
            launch_arguments={
                'camera_model': 'zedx',
                'camera_name': 'zed_front',
                # Verified 2026-04-24 via per-port enumeration:
                #   GMSL port 0 -> SN 42569280  (physical front, confirmed by user)
                #   GMSL port 1 -> SN 49910017
                #   GMSL port 2 -> SN 43779087
                'serial_number': '42569280',
                'publish_tf': 'false',           # robot_localization owns odom→base_link
                'publish_urdf': 'false',         # our URDF already includes zed_macro
                'ros_params_override_path': zed_front_config,
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_zed_front')),
        ),

        # ZED X Left (GMSL via ZED Link Quad)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('zed_wrapper'),
                '/launch/zed_camera.launch.py',
            ]),
            launch_arguments={
                'camera_model': 'zedx',
                'camera_name': 'zed_left',
                'serial_number': '43779087',
                'publish_tf': 'false',
                'publish_urdf': 'false',
                'ros_params_override_path': zed_left_config,
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_zed_left')),
        ),

        # ZED X Right (GMSL via ZED Link Quad)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('zed_wrapper'),
                '/launch/zed_camera.launch.py',
            ]),
            launch_arguments={
                'camera_model': 'zedx',
                'camera_name': 'zed_right',
                # TODO: confirm right-camera serial (was 42569280, now repurposed to front).
                'serial_number': '49910017',
                'publish_tf': 'false',
                'publish_urdf': 'false',
                'ros_params_override_path': zed_right_config,
            }.items(),
            condition=IfCondition(LaunchConfiguration('enable_zed_right')),
        ),

        # Xsens MTi-680G IMU/GNSS
        # Package: xsens_mti_ros2_driver (built from source via avros.repos)
        # Fork: https://github.com/Paarseus/Xsens_MTi_ROS_Driver_and_Ntrip_Client (ros2 branch)
        Node(
            package='xsens_mti_ros2_driver',
            executable='xsens_mti_node',
            name='xsens_mti_node',
            parameters=[xsens_config],
            output='screen',
            respawn=True,
            respawn_delay=5.0,
        ),

        # NTRIP client for RTK corrections (GPGGA -> caster -> RTCM3)
        # Package: ntrip (same fork as xsens_mti_ros2_driver — monorepo)
        Node(
            package='ntrip',
            executable='ntrip',
            name='ntrip_client',
            parameters=[ntrip_config],
            remappings=[
                ('nmea', '/nmea'),
                ('rtcm', '/rtcm'),
            ],
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_ntrip')),
        ),
    ])
