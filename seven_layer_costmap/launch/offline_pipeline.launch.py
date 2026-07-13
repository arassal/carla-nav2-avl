"""Exercise the real seven-layer nodes with synthetic ZED topics and odometry."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('seven_layer_costmap'),
                          'config', 'seven_layer_costmap.yaml')
    system_time = {'use_sim_time': False}
    return LaunchDescription([
        Node(package='seven_layer_costmap', executable='synthetic_zed',
             parameters=[system_time], output='screen'),
        Node(package='seven_layer_costmap', executable='three_zed_perception',
             parameters=[params, system_time], output='screen'),
        Node(package='seven_layer_costmap', executable='road_condition',
             parameters=[params, system_time], output='screen'),
        Node(package='seven_layer_costmap', executable='costmap_fusion',
             parameters=[params, system_time], output='screen'),
    ])
