"""Open the prepared top-down seven-layer RViz view."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('seven_layer_costmap'),
                          'config', 'seven_layer_costmap.rviz')
    return LaunchDescription([
        Node(package='rviz2', executable='rviz2', name='seven_layer_costmap_rviz',
             arguments=['-d', config], output='screen')
    ])
