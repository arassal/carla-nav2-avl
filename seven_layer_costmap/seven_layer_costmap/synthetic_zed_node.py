"""Synthetic three-ZED topic source for ROS integration tests without hardware/SVO."""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class SyntheticZedNode(Node):
    def __init__(self):
        super().__init__('synthetic_three_zed')
        self.declare_parameter('camera_names', ['zed_front', 'zed_left', 'zed_right'])
        self.declare_parameter('frequency', 10.0)
        self.declare_parameter('width', 160)
        self.declare_parameter('height', 90)
        self.declare_parameter('moving_vehicle', True)
        self._bridge = CvBridge()
        self._frame = 0
        self._camera_publishers = {}
        for camera in self.get_parameter('camera_names').value:
            prefix = f'/{camera}/{camera}_node'
            self._camera_publishers[camera] = {
                'rgb': self.create_publisher(Image, prefix + '/left/color/rect/image',
                                             qos_profile_sensor_data),
                'depth': self.create_publisher(Image, prefix + '/depth/depth_registered',
                                               qos_profile_sensor_data),
                'info': self.create_publisher(CameraInfo, prefix + '/left/camera_info',
                                              qos_profile_sensor_data),
            }
        front = self.get_parameter('camera_names').value[0]
        self._odom_pub = self.create_publisher(
            Odometry, f'/{front}/{front}_node/odom', qos_profile_sensor_data)
        self.create_timer(1.0 / float(self.get_parameter('frequency').value), self._publish)

    def _images(self, index):
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        rgb = np.full((height, width, 3), (70, 85, 95), dtype=np.uint8)
        # Bright lane markings and a small red signal exercise visual layers.
        rgb[height // 2:, width // 3:width // 3 + 3] = 220
        rgb[height // 2:, 2 * width // 3:2 * width // 3 + 3] = 220
        if (self._frame // 50) % 2 == 0 and index == 0:
            rgb[5:12, width // 2 - 3:width // 2 + 4] = (0, 0, 255)
        depth = np.full((height, width), 25.0, dtype=np.float32)
        # Static obstacle plus laterally moving object.
        depth[height // 2 - 8:height // 2 + 8, width // 2 - 10:width // 2 + 10] = 8.0
        moving_col = int(width * 0.25 + (self._frame % 40) * width * 0.01)
        depth[height // 2:height // 2 + 10,
              max(0, moving_col - 4):min(width, moving_col + 4)] = 6.0
        return rgb, depth

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        width = int(self.get_parameter('width').value)
        height = int(self.get_parameter('height').value)
        fx = width * 0.8
        fy = width * 0.8
        for index, (camera, publishers) in enumerate(self._camera_publishers.items()):
            rgb, depth = self._images(index)
            rgb_msg = self._bridge.cv2_to_imgmsg(rgb, 'bgr8')
            depth_msg = self._bridge.cv2_to_imgmsg(depth, '32FC1')
            rgb_msg.header.stamp = depth_msg.header.stamp = stamp
            rgb_msg.header.frame_id = depth_msg.header.frame_id = camera + '_left_camera_optical_frame'
            info = CameraInfo()
            info.header = rgb_msg.header
            info.width, info.height = width, height
            info.k = [fx, 0.0, width / 2, 0.0, fy, height / 2, 0.0, 0.0, 1.0]
            publishers['rgb'].publish(rgb_msg)
            publishers['depth'].publish(depth_msg)
            publishers['info'].publish(info)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        if bool(self.get_parameter('moving_vehicle').value):
            odom.pose.pose.position.x = self._frame * 0.02
        yaw = 0.03 * math.sin(self._frame * 0.02)
        odom.pose.pose.orientation.z = math.sin(yaw / 2)
        odom.pose.pose.orientation.w = math.cos(yaw / 2)
        self._odom_pub.publish(odom)
        self._frame += 1


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticZedNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
