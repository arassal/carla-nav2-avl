#!/usr/bin/env python3
"""Publish a synthetic /perception/costmap so the downstream chain can be
exercised with no cameras, no lidar, and no Jetson.

Why this exists: the autodrive preflight in `campus_navigator.py` refuses a
destination click while `/perception/costmap_cloud` is stale, and that topic is
produced by `deploy/costmap_to_cloud.py` from `/perception/costmap`. Debugging
anything in that chain otherwise needs the whole sensor stack. This stands in
for the perception node so the bridge, the Nav2 ObstacleLayer wiring, and the
navigator's staleness gates can be tested on a laptop.

    ros2 run ... # not installed as an entry point on purpose -- run directly:
    PYTHONPATH=. python3 tools/fake_costmap_publisher.py --obstacle-x 5.0

Then, in another shell:
    PYTHONPATH=. python3 deploy/costmap_to_cloud.py
    ros2 topic hz /perception/costmap_cloud

This is a TEST FIXTURE. It is not part of the runtime stack and must never be
launched on the car.
"""
import argparse

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

from perception_costmap.occupancy import (
    GridSpec, build_cost_array, to_occupancy_grid_msg)


class FakeCostmap(Node):
    def __init__(self, args):
        super().__init__("fake_costmap_publisher")
        self.grid = GridSpec(frame_id=args.frame_id)
        self.args = args
        self.pub = self.create_publisher(
            OccupancyGrid, "/perception/costmap", 1)
        self.create_timer(1.0 / args.rate, self._tick)
        self.get_logger().warning(
            "publishing SYNTHETIC costmap on /perception/costmap at %.1f Hz "
            "-- test fixture, never run this on the car" % args.rate)

    def _cost(self):
        g = self.grid
        shape = (g.height, g.width)
        # a drivable corridor straight ahead, everything else off-road
        road = np.zeros(shape, dtype=bool)
        half = int(round(self.args.road_half_width / g.resolution))
        centre_row = int(np.floor((0.0 - g.y_min) / g.resolution))
        road[centre_row - half:centre_row + half + 1, :] = True

        obstacle = np.zeros(shape, dtype=bool)
        if self.args.obstacle_x is not None:
            col = int(np.floor((self.args.obstacle_x - g.x_min) / g.resolution))
            row = int(np.floor((self.args.obstacle_y - g.y_min) / g.resolution))
            r = max(1, int(round(0.3 / g.resolution)))
            obstacle[row - r:row + r + 1, col - r:col + r + 1] = True

        return build_cost_array(
            g, road, obstacle, known_mask=np.ones(shape, dtype=bool),
            offroad_cost=self.args.offroad_cost)

    def _tick(self):
        msg = to_occupancy_grid_msg(
            self._cost(), self.grid, stamp=self.get_clock().now().to_msg())
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--frame-id", default="base_link")
    ap.add_argument("--road-half-width", type=float, default=2.0,
                    help="metres of drivable corridor either side of centre")
    ap.add_argument("--obstacle-x", type=float, default=5.0,
                    help="metres ahead; omit with --no-obstacle")
    ap.add_argument("--obstacle-y", type=float, default=0.0)
    ap.add_argument("--no-obstacle", action="store_true")
    ap.add_argument("--offroad-cost", type=int, default=97,
                    help="97 matches perception_dinosaur.yaml (the car config)")
    args = ap.parse_args()
    if args.no_obstacle:
        args.obstacle_x = None

    rclpy.init()
    node = FakeCostmap(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
