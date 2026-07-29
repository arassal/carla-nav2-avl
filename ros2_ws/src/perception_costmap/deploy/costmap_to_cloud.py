#!/usr/bin/env python3
"""Publish /perception/costmap's obstacle cells as a PointCloud2 for Nav2's
ObstacleLayer.

Why not StaticLayer (two failed attempts before this):
  1. Fed the raw base_link grid -> StaticLayer places a robot-relative map
     once and never re-places it as the robot moves.
  2. Fed a grid whose origin was transformed into odom -> Costmap2D is
     axis-aligned and silently DISCARDS the origin's yaw, so every obstacle
     landed rotated by the robot's heading. Measured: perception had the
     obstacle at 1.37 m dead ahead while nav2 read cost=0 there.
  3. Resampling to an axis-aligned odom grid fixes the rotation, but the grid
     then changes size/origin every frame -- a StaticLayer is built for a
     FIXED map and thrashes on that.

ObstacleLayer is the layer meant for this: it consumes sensor observations in
a sensor frame, transforms them through TF every update, marks obstacles and
raytrace-clears free space. No rotation assumption, no fixed-map assumption.
So we hand the costmap's occupied cells over as a point cloud in base_link
and let ObstacleLayer do what it is designed to do.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

OBST_THRESH = 97      # >=97 is near-lethal (kerb/wall) or lethal in this stack
POINT_Z = 0.35        # inside obstacle_layer's [min_obstacle_height, max]


class CostmapToCloud(Node):
    def __init__(self):
        super().__init__("costmap_to_cloud")
        self.pub = self.create_publisher(PointCloud2, "/perception/costmap_cloud", 1)
        self.create_subscription(OccupancyGrid, "/perception/costmap",
                                 self.cb, qos_profile_sensor_data)
        self.n = 0

    def cb(self, msg):
        r = msg.info.resolution
        W, H = msg.info.width, msg.info.height
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.asarray(msg.data, dtype=np.int8).reshape(H, W)

        # Emit only the FIRST obstacle cell along each ray from the robot.
        #
        # The perception costmap marks the occlusion shadow BEHIND an obstacle
        # as lethal (100) -- identical to the obstacle itself. Measured with a
        # dummy at 5 m: 7523 lethal cells forming a solid wall from 5 m out to
        # 16 m spanning the whole lane, so any goal past it was unreachable and
        # the planner correctly returned an empty path. Dumping every occupied
        # cell into the cloud reproduces that wall inside nav2.
        #
        # Ray-casting and keeping only the nearest hit per bearing yields the
        # visible surface of obstacles. Space behind them is simply never
        # marked, so it stays free/unknown and the planner can route around --
        # which is the whole point. Anything genuinely there gets marked as
        # soon as the robot moves and actually observes it.
        A = np.radians(np.arange(-100.0, 100.5, 0.5))
        R = np.arange(0.3, 15.0, 0.05)
        X = np.cos(A)[:, None] * R[None, :]
        Y = np.sin(A)[:, None] * R[None, :]
        ix = ((X - ox) / r).astype(np.int32)
        iy = ((Y - oy) / r).astype(np.int32)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        v = np.full(X.shape, -1, dtype=np.int16)
        v[ok] = grid[iy[ok], ix[ok]]

        hit = v >= OBST_THRESH
        has_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        rows = np.nonzero(has_hit)[0]
        if len(rows) == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            cols = first[rows]
            x = X[rows, cols].astype(np.float32)
            y = Y[rows, cols].astype(np.float32)
            z = np.full_like(x, POINT_Z)
            pts = np.stack([x, y, z], axis=1)

        header = msg.header          # base_link, same stamp -- TF handles the rest
        cloud = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.pub.publish(cloud)

        self.n += 1
        if self.n % 20 == 0:
            self.get_logger().info(
                f"published {self.n} clouds, latest {len(pts)} obstacle points "
                f"(frame={header.frame_id})")


def main():
    rclpy.init()
    node = CostmapToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
