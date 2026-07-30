"""perception_costmap_v2 — reimplementation of the camera+lidar -> Nav2
costmap pipeline (see ../DESIGN.md). Core modules stay ROS-free / CARLA-free;
only costmap_node.py imports rclpy, only tools/carla_feed.py imports carla.
"""
