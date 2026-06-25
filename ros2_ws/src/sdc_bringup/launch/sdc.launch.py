import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('sdc_bringup'),
                       'config', 'params.yaml')
    rviz = os.path.join(get_package_share_directory('sdc_bringup'),
                        'config', 'sdc.rviz')
    ws = '/home/merabro/selfdrive_carla_ue5/ros2_ws/src'

    world = ExecuteProcess(
        cmd=['/usr/bin/python3.11',
             f'{ws}/world_setup/world_setup/world_setup.py'],
        output='screen')
    planner = Node(package='route_planner', executable='route_planner_node',
                   output='screen')
    controller = Node(package='controller', executable='controller_node',
                      parameters=[cfg], output='screen')
    guard = Node(package='collision_guard', executable='collision_guard_node',
                 parameters=[cfg], output='screen')
    rviz_proc = ExecuteProcess(cmd=['rviz2', '-d', rviz], output='screen')

    return LaunchDescription([
        world,
        TimerAction(period=8.0, actions=[planner]),
        TimerAction(period=9.0, actions=[controller, guard, rviz_proc]),
    ])
