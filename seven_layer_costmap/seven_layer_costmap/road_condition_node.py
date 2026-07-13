"""Camera-image road-condition baseline producing a costmap layer and status."""

import threading
import time
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from .core import GridSpec, infer_road_condition
from .perception import stamp_seconds


class RoadConditionNode(Node):
    def __init__(self):
        super().__init__('road_condition_layer')
        self.declare_parameter('camera_topics', [
            '/zed_front/zed_front_node/left/color/rect/image',
            '/zed_left/zed_left_node/left/color/rect/image',
            '/zed_right/zed_right_node/left/color/rect/image'])
        self.declare_parameter('width_m', 60.0)
        self.declare_parameter('height_m', 60.0)
        self.declare_parameter('resolution', 0.20)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('publish_frequency', 5.0)
        self.declare_parameter('stale_timeout_s', 0.5)
        self.declare_parameter('require_all_cameras', True)
        self.declare_parameter('max_camera_skew_s', 0.050)
        self.declare_parameter('timestamp_offsets_s.front', 0.0)
        self.declare_parameter('timestamp_offsets_s.left', 0.0)
        self.declare_parameter('timestamp_offsets_s.right', 0.0)
        self._bridge = CvBridge()
        self._results = {}
        self._received_wall = {}
        self._stamps = {}
        self._lock = threading.Lock()
        topics = list(self.get_parameter('camera_topics').value)
        logical_names = ('front', 'left', 'right')
        if len(topics) != len(logical_names):
            raise ValueError('road-condition node requires exactly three camera topics')
        offsets = [float(self.get_parameter(f'timestamp_offsets_s.{name}').value)
                   for name in logical_names]
        self._offsets = dict(zip(topics, offsets))
        for topic in topics:
            self.create_subscription(Image, topic,
                                     lambda msg, t=topic: self._image(msg, t),
                                     qos_profile_sensor_data)
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
                self._received_wall[topic] = time.monotonic()
                self._stamps[topic] = stamp_seconds(msg.header.stamp) + self._offsets[topic]
        except Exception as error:
            self.get_logger().warn(f'{topic}: {error}', throttle_duration_sec=5.0)

    def _publish(self):
        spec = GridSpec(float(self.get_parameter('width_m').value),
                        float(self.get_parameter('height_m').value),
                        float(self.get_parameter('resolution').value))
        with self._lock:
            now = time.monotonic()
            fresh_topics = [topic for topic, received in self._received_wall.items()
                            if now - received <= float(self.get_parameter('stale_timeout_s').value)]
            results = [self._results[topic] for topic in fresh_topics]
            stamps = [self._stamps[topic] for topic in fresh_topics]
        required = len(self.get_parameter('camera_topics').value)
        if not results or (bool(self.get_parameter('require_all_cameras').value) and
                           len(fresh_topics) != required):
            self._status_pub.publish(String(
                data=f'NOT_READY:fresh_cameras={len(fresh_topics)}/{required}'))
            return
        skew = max(stamps) - min(stamps)
        if skew > float(self.get_parameter('max_camera_skew_s').value):
            self._status_pub.publish(String(data=f'NOT_READY:skew_ms={skew * 1000:.1f}'))
            return
        condition, confidence, cost = max(results, key=lambda item: item[2])
        grid = np.zeros(spec.shape, dtype=np.int8)
        # Apply the inferred traction/visibility penalty only in the forward drivable half.
        grid[:, spec.shape[1] // 2:] = cost
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
        self._status_pub.publish(String(
            data=f'{condition}:{confidence:.3f}:cost={cost}:skew_ms={skew * 1000:.1f}'))


def main(args=None):
    rclpy.init(args=args)
    node = RoadConditionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
