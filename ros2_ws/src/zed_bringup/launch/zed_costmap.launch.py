"""
Launch: ZED X cameras + costmap + RViz2

Assumes the three ZED X ROS2 wrappers are already running externally:
  ros2 launch zed_wrapper zed_camera.launch.py camera_name:=zed_front ...
  ros2 launch zed_wrapper zed_camera.launch.py camera_name:=zed_left  ...
  ros2 launch zed_wrapper zed_camera.launch.py camera_name:=zed_right ...

This launch file starts:
  1. Static TF: map → odom
  2. Static TF: odom → base_link  (identity until real odometry is wired in)
  3. Static TFs: base_link → each camera optical frame
  4. zed_bridge_node  (depth + YOLO → /scan + /ped_scan)
  5. costmap_node     (LaserScan → OccupancyGrid, unchanged from CARLA version)
  6. RViz2

Camera mount positions — edit to match your vehicle rig.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('zed_bringup')
    rviz_config = os.path.join(bringup_dir, 'config', 'rviz_config.rviz')

    # ── Global TF skeleton ────────────────────────────────────────────────────
    map_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='map_to_odom_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--frame-id', 'map', '--child-frame-id', 'odom'],
        output='screen',
    )

    # Identity odom→base_link until real odometry is integrated.
    # Replace with your odometry source (wheel encoders, GPS, VIO, etc.)
    odom_to_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='odom_to_base_tf',
        arguments=['--x', '0', '--y', '0', '--z', '0',
                   '--frame-id', 'odom', '--child-frame-id', 'base_link'],
        output='screen',
    )

    # ── Camera mount TFs (base_link → ZED optical frames) ────────────────────
    # ZED optical frame: z=forward, x=right, y=down
    # Rotation from base_link (x=fwd, y=left, z=up) to ZED front optical:
    #   roll=−90°, yaw=−90°  → quaternion (−0.5, 0.5, −0.5, 0.5)
    cam_front_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='cam_front_tf',
        arguments=['--x', '2.0', '--y', '0.0', '--z', '1.2',
                   '--qx', '-0.5', '--qy', '0.5',
                   '--qz', '-0.5', '--qw', '0.5',
                   '--frame-id', 'base_link',
                   '--child-frame-id', 'zed_front_left_camera_optical_frame'],
        output='screen',
    )

    # Left camera: 0.5 m fwd, 0.9 m left, 1.2 m up, yaw +90° then optical rotation
    cam_left_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='cam_left_tf',
        arguments=['--x', '0.5', '--y', '0.9', '--z', '1.2',
                   '--qx', '-0.5', '--qy', '-0.5',
                   '--qz',  '0.5', '--qw',  '0.5',
                   '--frame-id', 'base_link',
                   '--child-frame-id', 'zed_left_left_camera_optical_frame'],
        output='screen',
    )

    # Right camera: 0.5 m fwd, 0.9 m right, 1.2 m up, yaw -90° then optical rotation
    cam_right_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='cam_right_tf',
        arguments=['--x', '0.5', '--y', '-0.9', '--z', '1.2',
                   '--qx', '-0.5', '--qy', '0.5',
                   '--qz',  '0.5', '--qw', '-0.5',
                   '--frame-id', 'base_link',
                   '--child-frame-id', 'zed_right_left_camera_optical_frame'],
        output='screen',
    )

    # ── ZED bridge ────────────────────────────────────────────────────────────
    zed_bridge = Node(
        package='zed_bridge',
        executable='zed_bridge_node',
        name='zed_bridge',
        output='screen',
        emulate_tty=True,
    )

    # ── Costmap — 5 s delay so TF and camera topics are up ───────────────────
    costmap = TimerAction(
        period=5.0,
        actions=[Node(
            package='zed_bridge',
            executable='costmap_node',
            name='costmap_node',
            output='screen',
        )],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz = TimerAction(
        period=8.0,
        actions=[Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        )],
    )

    return LaunchDescription([
        map_to_odom,
        odom_to_base,
        cam_front_tf,
        cam_left_tf,
        cam_right_tf,
        zed_bridge,
        costmap,
        rviz,
    ])
