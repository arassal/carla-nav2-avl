#!/usr/bin/env python3
"""
CARLA Localization Node - Original implementation
Publishes odometry and TF transforms from CARLA ego state
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Transform, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
import carla
import numpy as np
from math import sin, cos


class CarlaLocalizationNode(Node):
    """Publishes ego vehicle odometry and transform tree"""

    def __init__(self):
        super().__init__("carla_localization")

        self.carla_client = carla.Client("localhost", 2000)
        self.carla_client.set_timeout(10.0)
        self.world = self.carla_client.get_world()

        # Get ego vehicle
        actors = self.world.get_actors()
        self.ego = None
        for actor in actors:
            if actor.type_id.startswith("vehicle."):
                self.ego = actor
                break

        if self.ego is None:
            self.get_logger().error("No ego vehicle found in CARLA world")
            return

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer
        self.create_timer(0.05, self.publish_odometry)  # 20 Hz

        self.get_logger().info("Localization node started")

    def publish_odometry(self):
        """Publish odometry from CARLA ego state"""
        if self.ego is None:
            return

        # Get ego state
        loc = self.ego.get_location()
        rot = self.ego.get_transform().rotation
        vel = self.ego.get_velocity()
        ang_vel = self.ego.get_angular_velocity()

        # Create odometry message
        odom = Odometry()
        odom.header.frame_id = "odom"
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.child_frame_id = "ego"

        # Position
        odom.pose.pose.position.x = float(loc.x)
        odom.pose.pose.position.y = float(loc.y)
        odom.pose.pose.position.z = float(loc.z)

        # Orientation (convert yaw to quaternion)
        yaw = np.radians(rot.yaw)
        odom.pose.pose.orientation.z = sin(yaw / 2)
        odom.pose.pose.orientation.w = cos(yaw / 2)

        # Velocity
        speed = np.sqrt(vel.x ** 2 + vel.y ** 2)
        odom.twist.twist.linear.x = float(speed)
        odom.twist.twist.angular.z = float(ang_vel.z)

        self.odom_pub.publish(odom)

        # Publish TF transform
        tf = TransformStamped()
        tf.header.frame_id = "odom"
        tf.header.stamp = odom.header.stamp
        tf.child_frame_id = "ego"

        tf.transform.translation.x = float(loc.x)
        tf.transform.translation.y = float(loc.y)
        tf.transform.translation.z = float(loc.z)

        tf.transform.rotation = odom.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaLocalizationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
