"""Dependency-light end-to-end verification with seven deterministic layer sources."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('seven_layer_costmap'),
                          'config', 'seven_layer_costmap.yaml')
    return LaunchDescription([
        Node(package='seven_layer_costmap', executable='synthetic_layers',
             parameters=[params, {'use_sim_time': False, 'publish_road_condition': True}],
             output='screen'),
        Node(package='seven_layer_costmap', executable='costmap_fusion',
             parameters=[params, {'use_sim_time': False}], output='screen'),
    ])
