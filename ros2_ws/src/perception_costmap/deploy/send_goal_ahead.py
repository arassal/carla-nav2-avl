#!/usr/bin/env python3
"""Send a NavigateToPose goal N meters straight ahead of the robot's CURRENT
pose, expressed in the odom frame.

The real car has no map frame (no SLAM/GPS anchor), so goals must be given in
odom -- and odom's origin is wherever the EKF started, NOT the robot. So a
"20 feet ahead" goal has to be computed from the live odom->base_link TF at
send time, not hardcoded. Usage:

    send_goal_ahead.py [meters]      # default 6.096 m == 20 ft
"""
import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
import tf2_ros


def main():
    dist = float(sys.argv[1]) if len(sys.argv) > 1 else 6.096

    rclpy.init()
    node = Node("goal_ahead_sender")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    # Spin briefly so the TF buffer actually fills before we look up.
    tf = None
    for _ in range(50):
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tf = buf.lookup_transform("odom", "base_link", rclpy.time.Time())
            break
        except Exception:
            continue
    if tf is None:
        print("[goal] FAILED: no odom->base_link TF available", flush=True)
        return

    t = tf.transform.translation
    q = tf.transform.rotation
    yaw = 2.0 * math.atan2(q.z, q.w)
    gx = t.x + dist * math.cos(yaw)
    gy = t.y + dist * math.sin(yaw)

    print(f"[goal] robot now at odom ({t.x:.2f}, {t.y:.2f}) heading "
          f"{math.degrees(yaw):.1f}deg", flush=True)
    print(f"[goal] goal = {dist:.2f} m ahead -> odom ({gx:.2f}, {gy:.2f})",
          flush=True)

    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    if not client.wait_for_server(timeout_sec=15.0):
        print("[goal] FAILED: navigate_to_pose action server not up", flush=True)
        return

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "odom"
    goal.pose.header.stamp = node.get_clock().now().to_msg()
    goal.pose.pose.position.x = gx
    goal.pose.pose.position.y = gy
    # keep the current heading as the goal heading (drive straight ahead)
    goal.pose.pose.orientation.z = q.z
    goal.pose.pose.orientation.w = q.w

    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    handle = future.result()
    if handle is None or not handle.accepted:
        print("[goal] REJECTED or no response", flush=True)
        return
    print("[goal] ACCEPTED -- driving", flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
