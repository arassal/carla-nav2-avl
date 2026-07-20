"""Exercise the real camera-costmap nodes with synthetic ZED topics."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('zedx_vision_costmap'),
                          'config', 'zedx_vision_costmap.yaml')
    system_time = {'use_sim_time': False}
    return LaunchDescription([
        Node(package='zedx_vision_costmap', executable='synthetic_zed',
             parameters=[system_time], output='screen'),
        Node(package='zedx_vision_costmap', executable='three_zed_perception',
             parameters=[params, system_time], output='screen'),
        Node(package='zedx_vision_costmap', executable='costmap_fusion',
             parameters=[params, system_time], output='screen'),
    ])
