#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point


class ConeDetectorNode(Node):
    def __init__(self):
        super().__init__('cone_detector')

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.center_pub = self.create_publisher(Point, '/cone/center', 10)
        self.debug_pub = self.create_publisher(Image, '/cone/debug_image', 10)

        self.get_logger().info('Cone detector node started')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_orange = (5, 100, 100)
        upper_orange = (25, 255, 255)

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 300:
                x, y, w, h = cv2.boundingRect(largest)

                cx = x + w / 2.0
                cy = y + h / 2.0

                point = Point()
                point.x = cx
                point.y = cy
                point.z = area
                self.center_pub.publish(point)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

        debug_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()