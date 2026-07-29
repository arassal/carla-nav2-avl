"""
Bring up the perception costmap node.

    ros2 launch perception_costmap perception.launch.py
    ros2 launch perception_costmap perception.launch.py lidar_topic:=/carla/lidar rviz:=true

Paths are resolved from the installed package share dir -- no hardcoded
home directories. The lidar topic is a launch arg so the same launch works
for CARLA and for the real car. Camera topics are NOT launch args: with
multi-camera BEV fusion (Task 6) each camera has its own YAML block
(cameras: [...], then a per-name block with image_topic/camera_info_topic/
mounting) in config/perception_costmap.yaml -- edit that file (or pass an
override config) to point cameras at the right topics.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('perception_costmap')
    default_cfg = os.path.join(pkg, 'config', 'perception_costmap.yaml')

    cfg = LaunchConfiguration('config')
    lidar_topic = LaunchConfiguration('lidar_topic')
    use_rviz = LaunchConfiguration('rviz')

    args = [
        DeclareLaunchArgument('config', default_value=default_cfg,
                              description='perception_costmap params YAML'),
        DeclareLaunchArgument('lidar_topic', default_value='/lidar/points'),
        DeclareLaunchArgument('rviz', default_value='false'),
    ]

    perception = Node(
        package='perception_costmap',
        executable='costmap_node',
        name='perception_costmap',
        output='screen',
        parameters=[cfg],
        remappings=[
            ('/lidar/points', lidar_topic),
        ],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen', condition=IfCondition(use_rviz),
    )

    return LaunchDescription(args + [perception, rviz])
