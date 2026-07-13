"""Deterministic CARLA/milestone harness for layers whose real producers are pending."""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node

from .core import GridSpec, LAYER_NAMES, inflate, rasterize_predictions


class SyntheticLayersNode(Node):
    """Publishes observable test layers; must not be enabled for vehicle operation."""
    def __init__(self):
        super().__init__('synthetic_seven_layer_sources')
        self.declare_parameter('width_m', 60.0)
        self.declare_parameter('height_m', 60.0)
        self.declare_parameter('resolution', 0.20)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('publish_road_condition', False)
        self._pubs = {name: self.create_publisher(
            OccupancyGrid, f'/seven_layer_costmap/layers/{name}', 1)
            for name in LAYER_NAMES
            if name != 'road_condition' or self.get_parameter('publish_road_condition').value}
        self.create_timer(0.2, self._publish)

    def _message(self, grid):
        spec = self.spec
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.info.resolution = spec.resolution
        msg.info.width, msg.info.height = spec.shape[1], spec.shape[0]
        msg.info.origin.position.x, msg.info.origin.position.y = (-spec.width_m / 2,
                                                                  -spec.height_m / 2)
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.astype(np.int8).ravel().tolist()
        return msg

    @property
    def spec(self):
        return GridSpec(float(self.get_parameter('width_m').value),
                        float(self.get_parameter('height_m').value),
                        float(self.get_parameter('resolution').value))

    def _publish(self):
        spec = self.spec
        h, w = spec.shape
        layers = {name: np.zeros(spec.shape, dtype=np.uint8) for name in self._pubs}
        # A low-cost lane departure field with a 6 m center corridor.
        layers['lanelet'][:h // 2 - 15, w // 2:] = 80
        layers['lanelet'][h // 2 + 15:, w // 2:] = 80
        layers['lanelet'][:, :w // 2] = 95
        # Static wall and transient obstacle samples for end-to-end validation.
        layers['static_obstacle'][h // 2 + 45:h // 2 + 50, w // 2 - 20:w // 2 + 20] = 100
        layers['spatio_temporal_voxel'][h // 2 + 20:h // 2 + 25, w // 2 + 10:w // 2 + 15] = 100
        layers['prediction'] = rasterize_predictions(spec, [(5.0, -2.0, 1.5, 0.5)])
        obstacle_union = np.maximum(layers['static_obstacle'], layers['spatio_temporal_voxel'])
        layers['inflation'] = inflate(obstacle_union, 2.5, spec.resolution, 1.2)
        layers['traffic_regulation'][h // 2 + 70:h // 2 + 73, w // 2 - 15:w // 2 + 15] = 100
        if 'road_condition' in layers:
            layers['road_condition'][:, w // 2:] = 20
        for name, publisher in self._pubs.items():
            publisher.publish(self._message(layers[name]))


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticLayersNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
