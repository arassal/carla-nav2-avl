#!/usr/bin/env python3
"""Render /perception/costmap as RViz points, avoiding RViz Map's texture shader.

Use this only as a visualization fallback. Nav2 continues to consume the
original OccupancyGrid unchanged.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker


class CostmapMarkerViz(Node):
    def __init__(self, smoothing_frames: int) -> None:
        super().__init__("costmap_marker_viz")
        self.history = deque(maxlen=max(1, smoothing_frames))
        self.grid_shape = None
        self.create_subscription(OccupancyGrid, "/perception/costmap", self._on_costmap, 1)
        self.publisher = self.create_publisher(Marker, "/perception/costmap_markers", 1)
        self.get_logger().info("Rendering /perception/costmap as /perception/costmap_markers")

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        shape = (msg.info.height, msg.info.width)
        if shape != self.grid_shape:
            self.history.clear()
            self.grid_shape = shape
        current = np.asarray(msg.data, dtype=np.int16).reshape(shape)
        self.history.append(current)
        values = current.copy()
        if len(self.history) >= 3:
            stack = np.stack(self.history)
            majority = len(self.history) // 2 + 1
            unknown_votes = (stack < 0).sum(axis=0)
            lethal_votes = (stack >= 99).sum(axis=0)
            free_votes = ((stack >= 0) & (stack < 99)).sum(axis=0)
            values[unknown_votes >= majority] = -1
            values[free_votes >= majority] = 0
            values[lethal_votes >= majority] = 100

        marker = Marker()
        marker.header = msg.header
        marker.ns = "perception_costmap"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = msg.info.resolution * 0.9
        marker.scale.y = msg.info.resolution * 0.9

        width = msg.info.width
        resolution = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        for index, value in enumerate(values.flat):
            if value < 0:
                continue
            row, col = divmod(index, width)
            marker.points.append(Point(
                x=ox + (col + 0.5) * resolution,
                y=oy + (row + 0.5) * resolution,
                z=0.02,
            ))
            if value >= 99:
                marker.colors.append(ColorRGBA(r=0.95, g=0.05, b=0.15, a=0.85))
            else:
                marker.colors.append(ColorRGBA(r=0.10, g=0.85, b=0.20, a=0.75))
        self.publisher.publish(marker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoothing-frames", type=int, default=3,
                        help="Odd-sized visual majority window")
    args = parser.parse_args()
    if args.smoothing_frames < 1 or args.smoothing_frames % 2 == 0:
        parser.error("--smoothing-frames must be a positive odd number")
    rclpy.init()
    node = CostmapMarkerViz(args.smoothing_frames)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
