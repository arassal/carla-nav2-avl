"""Seven-layer milestone demo: synthetic layers plus real/CARLA camera inference."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('seven_layer_costmap')
    params = os.path.join(share, 'config', 'seven_layer_costmap.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    common = {'use_sim_time': use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='seven_layer_costmap', executable='synthetic_layers',
             parameters=[params, common], output='screen'),
        Node(package='seven_layer_costmap', executable='road_condition',
             parameters=[params, common], output='screen'),
        Node(package='seven_layer_costmap', executable='costmap_fusion',
             parameters=[params, common], output='screen'),
    ])
