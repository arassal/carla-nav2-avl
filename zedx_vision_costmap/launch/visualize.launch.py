"""Open the prepared top-down zedx-vision RViz view."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(get_package_share_directory('zedx_vision_costmap'),
                          'config', 'zedx_vision_costmap.rviz')
    return LaunchDescription([
        Node(package='rviz2', executable='rviz2', name='zedx_vision_costmap_rviz',
             arguments=['-d', config], output='screen')
    ])
