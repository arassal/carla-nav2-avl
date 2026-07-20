"""Launch three physical ZED X cameras and the live vision costmap pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def live_instance(wrapper_launch, name, serial, override):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(wrapper_launch),
        launch_arguments={
            'camera_model': 'zedx',
            'camera_name': name,
            'node_name': name + '_node',
            'serial_number': serial,
            'svo_path': '',
            'ros_params_override_path': override,
            'publish_urdf': 'true',
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'publish_imu_tf': 'false',
            'use_sim_time': 'false',
        }.items())


def validate_inputs(context):
    serials = []
    for argument in ('front_serial', 'left_serial', 'right_serial'):
        raw = LaunchConfiguration(argument).perform(context)
        try:
            serial = int(raw)
        except ValueError as error:
            raise RuntimeError(f'{argument} must be a positive integer: {raw}') from error
        if serial <= 0:
            raise RuntimeError(f'{argument} must be a positive integer: {raw}')
        serials.append(serial)
    if len(set(serials)) != 3:
        raise RuntimeError('front, left, and right must use three distinct camera serials')
    override = LaunchConfiguration('zed_override_path').perform(context)
    if not os.path.isabs(override) or not os.path.isfile(override):
        raise RuntimeError(f'zed_override_path must be an absolute existing file: {override}')
    return []


def generate_launch_description():
    share = get_package_share_directory('zedx_vision_costmap')
    wrapper = os.path.join(get_package_share_directory('zed_wrapper'),
                           'launch', 'zed_camera.launch.py')
    params = os.path.join(share, 'config', 'zedx_vision_costmap.yaml')
    default_override = os.path.join(share, 'config', 'zed_live_override.yaml')
    rviz_config = os.path.join(share, 'config', 'zedx_vision_costmap.rviz')
    override = LaunchConfiguration('zed_override_path')
    front = LaunchConfiguration('front_serial')
    left = LaunchConfiguration('left_serial')
    right = LaunchConfiguration('right_serial')
    return LaunchDescription([
        DeclareLaunchArgument('front_serial', default_value='42569280',
                              description='Physical front ZED X serial number'),
        DeclareLaunchArgument('left_serial', default_value='49910017',
                              description='Physical left ZED X serial number'),
        DeclareLaunchArgument('right_serial', default_value='43779087',
                              description='Physical right ZED X serial number'),
        DeclareLaunchArgument(
            'zed_override_path', default_value=default_override,
            description='Absolute ZED wrapper override YAML for live cameras'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Start the prepared live three-camera RViz view'),
        OpaqueFunction(function=validate_inputs),
        live_instance(wrapper, 'zed_front', front, override),
        live_instance(wrapper, 'zed_left', left, override),
        live_instance(wrapper, 'zed_right', right, override),
        Node(package='zedx_vision_costmap', executable='three_zed_perception',
             parameters=[params], output='screen'),
        Node(package='zedx_vision_costmap', executable='costmap_fusion',
             parameters=[params, {'use_sim_time': False}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='three_zedx_live_rviz',
             arguments=['-d', rviz_config], output='screen',
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
