"""Active seven-layer ROS 2 costmap fusion node."""

import threading
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from .core import LAYER_NAMES, fuse_layers, normalize_layer


class FusionNode(Node):
    def __init__(self):
        super().__init__('seven_layer_costmap')
        self.declare_parameter('publish_frequency', 5.0)
        self.declare_parameter('output_topic', '/seven_layer_costmap/costmap')
        self.declare_parameter('require_all_layers', True)
        self.declare_parameter('stale_timeout_s', 1.0)
        for name in LAYER_NAMES:
            self.declare_parameter(f'weights.{name}', 1.0)
        self._layers = {}
        self._messages = {}
        self._received_s = {}
        self._lock = threading.Lock()
        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.VOLATILE)
        output_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for name in LAYER_NAMES:
            topic = f'/seven_layer_costmap/layers/{name}'
            self.create_subscription(OccupancyGrid, topic,
                                     lambda msg, n=name: self._receive(n, msg), input_qos)
        self._publisher = self.create_publisher(
            OccupancyGrid, self.get_parameter('output_topic').value, output_qos)
        hz = float(self.get_parameter('publish_frequency').value)
        self.create_timer(1.0 / hz, self._publish)
        self.get_logger().info('Waiting for seven costmap layers: ' + ', '.join(LAYER_NAMES))

    def _receive(self, name, msg):
        if (msg.info.width == 0 or msg.info.height == 0 or msg.info.resolution <= 0 or
                not msg.header.frame_id):
            self.get_logger().error(f'Rejected {name}: invalid grid metadata')
            return
        shape = (msg.info.height, msg.info.width)
        try:
            layer = normalize_layer(msg.data, shape)
        except (ValueError, TypeError) as error:
            self.get_logger().error(f'Rejected {name}: {error}')
            return
        with self._lock:
            if self._messages:
                reference = next(iter(self._messages.values()))
                origin = msg.info.origin
                reference_origin = reference.info.origin
                same_geometry = (msg.info.width == reference.info.width and
                                 msg.info.height == reference.info.height and
                                 abs(msg.info.resolution - reference.info.resolution) < 1e-6 and
                                 msg.header.frame_id == reference.header.frame_id and
                                 abs(origin.position.x - reference_origin.position.x) < 1e-6 and
                                 abs(origin.position.y - reference_origin.position.y) < 1e-6 and
                                 abs(origin.position.z - reference_origin.position.z) < 1e-6 and
                                 abs(origin.orientation.x - reference_origin.orientation.x) < 1e-6 and
                                 abs(origin.orientation.y - reference_origin.orientation.y) < 1e-6 and
                                 abs(origin.orientation.z - reference_origin.orientation.z) < 1e-6 and
                                 abs(origin.orientation.w - reference_origin.orientation.w) < 1e-6)
                if not same_geometry:
                    self.get_logger().error(f'Rejected {name}: grid geometry/frame mismatch')
                    return
            self._layers[name] = layer
            self._messages[name] = msg
            self._received_s[name] = self.get_clock().now().nanoseconds / 1e9

    def _publish(self):
        now = self.get_clock().now()
        now_s = now.nanoseconds / 1e9
        required = bool(self.get_parameter('require_all_layers').value)
        timeout = float(self.get_parameter('stale_timeout_s').value)
        with self._lock:
            missing = set(LAYER_NAMES) - set(self._layers)
            stale = [name for name, stamp in self._received_s.items() if now_s - stamp > timeout]
            if (required and missing) or stale:
                if missing:
                    self.get_logger().warn('Not publishing; missing: ' + ', '.join(sorted(missing)),
                                           throttle_duration_sec=5.0)
                if stale:
                    self.get_logger().warn('Not publishing; stale: ' + ', '.join(sorted(stale)),
                                           throttle_duration_sec=5.0)
                return
            weights = {name: float(self.get_parameter(f'weights.{name}').value)
                       for name in LAYER_NAMES}
            merged = fuse_layers(self._layers, weights)
            template = next(iter(self._messages.values()))
            output = OccupancyGrid()
            output.header = template.header
            output.header.stamp = now.to_msg()
            output.info = template.info
            output.data = merged.astype(np.int8).ravel().tolist()
        self._publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
