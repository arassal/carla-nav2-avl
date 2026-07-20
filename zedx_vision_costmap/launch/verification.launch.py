"""Dependency-light end-to-end verification with five deterministic layer sources."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('zedx_vision_costmap'),
                          'config', 'zedx_vision_costmap.yaml')
    return LaunchDescription([
        Node(package='zedx_vision_costmap', executable='synthetic_layers',
             parameters=[params, {'use_sim_time': False}],
             output='screen'),
        Node(package='zedx_vision_costmap', executable='costmap_fusion',
             parameters=[params, {'use_sim_time': False}], output='screen'),
    ])
