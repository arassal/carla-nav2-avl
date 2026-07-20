"""Launch three SVO wrapper instances and the camera-derived costmap pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def zed_instance(wrapper_launch, name, path, override):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(wrapper_launch),
        launch_arguments={
            'camera_model': 'zedx',
            'camera_name': name,
            'node_name': name + '_node',
            'svo_path': path,
            'ros_params_override_path': override,
            # The wrapper needs its namespaced internal camera-frame statics for
            # rectification and depth. Vehicle mount transforms remain
            # project-owned because dynamic/map TF publication stays disabled.
            'publish_urdf': 'true',
            'publish_tf': 'false',
            'publish_map_tf': 'false',
            'publish_imu_tf': 'false',
            'use_sim_time': 'false',
        }.items())


def validate_inputs(context):
    paths = []
    for argument in ('front_svo', 'left_svo', 'right_svo'):
        path = LaunchConfiguration(argument).perform(context)
        if not os.path.isabs(path):
            raise RuntimeError(f'{argument} must be an absolute path: {path}')
        if not os.path.isfile(path):
            raise RuntimeError(f'{argument} does not exist or is not a file: {path}')
        if not path.lower().endswith(('.svo', '.svo2')):
            raise RuntimeError(f'{argument} must end in .svo or .svo2: {path}')
        paths.append(os.path.realpath(path))
    if len(set(paths)) != 3:
        raise RuntimeError('front_svo, left_svo, and right_svo must be three distinct files')
    override = LaunchConfiguration('zed_override_path').perform(context)
    if not os.path.isfile(override):
        raise RuntimeError(f'zed_override_path does not exist: {override}')
    return []


def generate_launch_description():
    share = get_package_share_directory('seven_layer_costmap')
    wrapper = os.path.join(get_package_share_directory('zed_wrapper'),
                           'launch', 'zed_camera.launch.py')
    params = os.path.join(share, 'config', 'seven_layer_costmap.yaml')
    default_override = os.path.join(share, 'config', 'zed_svo_override.yaml')
    rviz_config = os.path.join(share, 'config', 'seven_layer_costmap.rviz')
    override = LaunchConfiguration('zed_override_path')
    front = LaunchConfiguration('front_svo')
    left = LaunchConfiguration('left_svo')
    right = LaunchConfiguration('right_svo')
    return LaunchDescription([
        DeclareLaunchArgument('front_svo', description='Absolute path to front ZED X SVO'),
        DeclareLaunchArgument('left_svo', description='Absolute path to left ZED X SVO'),
        DeclareLaunchArgument('right_svo', description='Absolute path to right ZED X SVO'),
        DeclareLaunchArgument(
            'zed_override_path', default_value=default_override,
            description='ZED wrapper override YAML; defaults to no-frame-drop quality mode'),
        DeclareLaunchArgument('rviz', default_value='true',
                              description='Start the prepared three-camera RViz view'),
        OpaqueFunction(function=validate_inputs),
        zed_instance(wrapper, 'zed_front', front, override),
        zed_instance(wrapper, 'zed_left', left, override),
        zed_instance(wrapper, 'zed_right', right, override),
        Node(package='seven_layer_costmap', executable='three_zed_perception',
             parameters=[params], output='screen'),
        Node(package='seven_layer_costmap', executable='costmap_fusion',
             parameters=[params, {'use_sim_time': False}], output='screen'),
        Node(package='rviz2', executable='rviz2', name='three_zed_vision_rviz',
             arguments=['-d', rviz_config], output='screen',
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
