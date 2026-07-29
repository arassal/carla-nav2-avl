#!/usr/bin/env python3
"""Resample /perception/costmap (base_link-relative) into an axis-aligned
odom-frame grid for Nav2's StaticLayer.

TWO separate bugs are being corrected here; the second one is subtle and cost
us a near-miss on a person standing in the lane.

1. StaticLayer places a robot-relative grid only once. It is built for a
   world-anchored map and does not re-derive placement from a moving frame_id,
   so feeding it frame_id=base_link freezes the obstacles wherever the robot
   happened to be for the first message.

2. (The one that mattered.) It is NOT enough to transform the grid's ORIGIN
   into odom and set the origin quaternion to the robot's heading.
   nav2_costmap_2d::Costmap2D is axis-aligned by construction -- it stores
   origin_x/origin_y and a resolution, and has no rotation term at all, so the
   origin's yaw is silently DISCARDED. A grid whose cells are laid out along
   base_link axes then gets stamped down along odom axes, and every obstacle
   lands rotated by the robot's heading error. Measured live: perception had
   1945 obstacle cells in the lane with the nearest at 1.37 m, while nav2's
   global costmap read cost=0 at 1.0-4.0 m dead ahead, and the planned path
   passed 0.01 m from a person. Perception was fine; the grid was rotated out
   from under it.

   So we must actually RESAMPLE: build an axis-aligned odom grid and, for each
   output cell, inverse-rotate its centre back into base_link to look up the
   source cell. Origin orientation is then left identity, which is the only
   thing Costmap2D can faithfully represent.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
import tf2_ros


class Republisher(Node):
    def __init__(self):
        super().__init__("costmap_odom_republisher")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pub = self.create_publisher(OccupancyGrid, "/perception/costmap_odom", 1)
        self.create_subscription(OccupancyGrid, "/perception/costmap",
                                 self.cb, qos_profile_sensor_data)
        self.n = 0

    def cb(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform(
                "odom", msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = 2.0 * math.atan2(q.z, q.w)
        c, s = math.cos(yaw), math.sin(yaw)

        r = msg.info.resolution
        W, H = msg.info.width, msg.info.height
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        src = np.asarray(msg.data, dtype=np.int8).reshape(H, W)

        # axis-aligned odom bounding box of the rotated source footprint
        cx = np.array([ox, ox + W * r, ox, ox + W * r])
        cy = np.array([oy, oy, oy + H * r, oy + H * r])
        wx = t.x + cx * c - cy * s
        wy = t.y + cx * s + cy * c
        minx, maxx = wx.min(), wx.max()
        miny, maxy = wy.min(), wy.max()

        out_w = int(math.ceil((maxx - minx) / r))
        out_h = int(math.ceil((maxy - miny) / r))

        # each output cell centre -> inverse-rotate into base_link -> source idx
        xs = minx + (np.arange(out_w) + 0.5) * r
        ys = miny + (np.arange(out_h) + 0.5) * r
        X, Y = np.meshgrid(xs, ys)
        dx, dy = X - t.x, Y - t.y
        bx = dx * c + dy * s
        by = -dx * s + dy * c

        ix = ((bx - ox) / r).astype(np.int32)
        iy = ((by - oy) / r).astype(np.int32)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

        out = np.full((out_h, out_w), -1, dtype=np.int8)   # -1 = unknown
        out[ok] = src[iy[ok], ix[ok]]

        g = OccupancyGrid()
        g.header.stamp = msg.header.stamp
        g.header.frame_id = "odom"
        g.info.resolution = r
        g.info.width = out_w
        g.info.height = out_h
        g.info.origin.position.x = float(minx)
        g.info.origin.position.y = float(miny)
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0   # identity: Costmap2D cannot rotate
        g.data = out.reshape(-1).tolist()
        self.pub.publish(g)

        self.n += 1
        if self.n % 20 == 0:
            self.get_logger().info(
                f"resampled {self.n} grids -> {out_w}x{out_h} axis-aligned, "
                f"origin=({minx:.2f},{miny:.2f}), robot yaw={math.degrees(yaw):.1f}deg, "
                f"occupied={(out >= 97).sum()}")


def main():
    rclpy.init()
    node = Republisher()
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
