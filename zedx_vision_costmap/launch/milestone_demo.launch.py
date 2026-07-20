"""Five-layer milestone demo using deterministic synthetic layer inputs."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('zedx_vision_costmap')
    params = os.path.join(share, 'config', 'zedx_vision_costmap.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    common = {'use_sim_time': use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(package='zedx_vision_costmap', executable='synthetic_layers',
             parameters=[params, common], output='screen'),
        Node(package='zedx_vision_costmap', executable='costmap_fusion',
             parameters=[params, common], output='screen'),
    ])
