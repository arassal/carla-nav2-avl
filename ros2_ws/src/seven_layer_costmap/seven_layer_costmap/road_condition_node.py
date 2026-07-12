"""Camera-image road-condition baseline producing a costmap layer and status."""

import threading
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .core import GridSpec, infer_road_condition


class RoadConditionNode(Node):
    def __init__(self):
        super().__init__('road_condition_layer')
        self.declare_parameter('camera_topics', [
            '/zed_front/rgb/image_rect_color', '/zed_left/rgb/image_rect_color',
            '/zed_right/rgb/image_rect_color'])
        self.declare_parameter('width_m', 60.0)
        self.declare_parameter('height_m', 60.0)
        self.declare_parameter('resolution', 0.20)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('publish_frequency', 5.0)
        self._bridge = CvBridge()
        self._results = {}
        self._lock = threading.Lock()
        for topic in self.get_parameter('camera_topics').value:
            self.create_subscription(Image, topic,
                                     lambda msg, t=topic: self._image(msg, t), 1)
        self._layer_pub = self.create_publisher(
            OccupancyGrid, '/seven_layer_costmap/layers/road_condition', 1)
        self._status_pub = self.create_publisher(String, '/road_condition/status', 10)
        hz = float(self.get_parameter('publish_frequency').value)
        self.create_timer(1.0 / hz, self._publish)

    def _image(self, msg, topic):
        try:
            result = infer_road_condition(self._bridge.imgmsg_to_cv2(msg, 'bgr8'))
            with self._lock:
                self._results[topic] = result
        except Exception as error:
            self.get_logger().warn(f'{topic}: {error}', throttle_duration_sec=5.0)

    def _publish(self):
        spec = GridSpec(float(self.get_parameter('width_m').value),
                        float(self.get_parameter('height_m').value),
                        float(self.get_parameter('resolution').value))
        with self._lock:
            results = list(self._results.values())
        if not results:
            return
        condition, confidence, cost = max(results, key=lambda item: item[2])
        grid = np.zeros(spec.shape, dtype=np.int8)
        # Apply the inferred traction/visibility penalty only in the forward drivable half.
        grid[spec.shape[0] // 2:, :] = cost
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.info.resolution = spec.resolution
        msg.info.width = spec.shape[1]
        msg.info.height = spec.shape[0]
        msg.info.origin.position.x = -spec.width_m / 2
        msg.info.origin.position.y = -spec.height_m / 2
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        self._layer_pub.publish(msg)
        self._status_pub.publish(String(data=f'{condition}:{confidence:.3f}:cost={cost}'))


def main(args=None):
    rclpy.init(args=args)
    node = RoadConditionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
