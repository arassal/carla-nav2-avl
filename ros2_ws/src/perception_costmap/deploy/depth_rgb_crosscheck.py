#!/usr/bin/env python3
"""Phase A: log-only cross-check of depth-derived vs RGB-derived obstacles.

Answers ONE question with real data before anything is wired into driving:
  "How often does camera depth flag a solid obstacle that the RGB pipeline
   (TwinLiteNet segmentation + YOLO + cones) does NOT already flag?"

Read-only. Subscribes only; publishes one optional viz topic. Touches nothing
in the costmap or control path. Per the 2026-07-16 plan, this must run across
real drive sessions BEFORE depth is merged into costmap_node -- the depth
pipeline has only ever been validated stationary, and this is how we learn
whether its signal is trustworthy without betting driving behavior on it.

Comparison is restricted to cells the RGB pipeline actually OBSERVED
(/perception/known). Comparing inside blind spots would be meaningless: those
cells carry an infilled guess, not an observation, so a "disagreement" there
says nothing about whether RGB missed something.

Interpreting the output:
  AGREE       depth says obstacle, RGB cost is already high  -> both caught it
  DEPTH-ONLY  depth says obstacle, RGB cost is low           -> RGB may have
              MISSED a real object (the interesting case -- this is what
              justifies integrating depth at all)
  RGB-ONLY    RGB cost high, depth sees nothing solid        -> expected and
              fine for flat things depth cannot see: painted lines, grass
              texture, off-road surface changes. NOT a depth failure.

A high DEPTH-ONLY rate = depth is adding real value.
A near-zero DEPTH-ONLY rate = depth is redundant with RGB; integrating it
would add risk/CPU for little gain, and the honest call would be to skip it.
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# RGB cost at/above this is "RGB already considers this dangerous".
# 50 sits above the blind-spot prior (25) and above the road-edge ramp's
# mid-range, but below off-road (97) and obstacle halos -- so it means
# "meaningfully elevated", not merely "not perfectly free".
RGB_HIGH_COST = 50


class DepthRgbCrosscheck(Node):
    def __init__(self, report_period):
        super().__init__('depth_rgb_crosscheck')
        self.depth_grid = None
        self.rgb_grid = None
        self.known = None

        self.create_subscription(OccupancyGrid, '/perception/depth_obstacle_grid',
                                 self._depth_cb, 1)
        self.create_subscription(OccupancyGrid, '/perception/costmap',
                                 self._rgb_cb, 1)
        self.create_subscription(OccupancyGrid, '/perception/known',
                                 self._known_cb, 1)
        # Cells where depth found something RGB did not -- the whole point.
        self.viz_pub = self.create_publisher(PointCloud2, '/perception/depth_only_obstacles', 1)

        self.samples = 0
        self.agree_total = 0
        self.depth_only_total = 0
        self.rgb_only_total = 0
        self.depth_only_frames = 0       # frames with ANY depth-only cell
        self.nearest_depth_only = []     # closest depth-only cell per frame (m)

        self.create_timer(0.2, self._compare)
        self.create_timer(report_period, self._report)
        self.get_logger().info(
            'depth_rgb_crosscheck up (READ-ONLY). Comparing depth vs RGB obstacles '
            f'in observed cells only; RGB_HIGH_COST={RGB_HIGH_COST}. '
            f'Reporting every {report_period:.0f}s.')

    def _depth_cb(self, msg):
        self.depth_grid = (msg, np.array(msg.data, np.int16).reshape(
            msg.info.height, msg.info.width))

    def _rgb_cb(self, msg):
        self.rgb_grid = np.array(msg.data, np.int16).reshape(
            msg.info.height, msg.info.width)

    def _known_cb(self, msg):
        self.known = np.array(msg.data, np.int8).reshape(
            msg.info.height, msg.info.width) > 0

    def _compare(self):
        if self.depth_grid is None or self.rgb_grid is None or self.known is None:
            return
        dmsg, dgrid = self.depth_grid
        if dgrid.shape != self.rgb_grid.shape or dgrid.shape != self.known.shape:
            return

        depth_obs = (dgrid >= 100) & self.known
        rgb_high = (self.rgb_grid >= RGB_HIGH_COST) & self.known

        agree = depth_obs & rgb_high
        depth_only = depth_obs & ~rgb_high
        rgb_only = rgb_high & ~depth_obs

        self.samples += 1
        self.agree_total += int(agree.sum())
        self.depth_only_total += int(depth_only.sum())
        self.rgb_only_total += int(rgb_only.sum())

        n_depth_only = int(depth_only.sum())
        if n_depth_only > 0:
            self.depth_only_frames += 1
            ys, xs = np.nonzero(depth_only)
            info = dmsg.info
            wx = info.origin.position.x + (xs + 0.5) * info.resolution
            wy = info.origin.position.y + (ys + 0.5) * info.resolution
            dist = np.hypot(wx, wy)
            self.nearest_depth_only.append(float(dist.min()))
            self._publish_viz(wx, wy, dmsg.header)

    def _publish_viz(self, wx, wy, header):
        pts = np.column_stack((wx, wy, np.zeros_like(wx))).astype(np.float32)
        msg = PointCloud2()
        msg.header = Header(stamp=header.stamp, frame_id=header.frame_id)
        msg.height = 1
        msg.width = pts.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.data = pts.tobytes()
        self.viz_pub.publish(msg)

    def _report(self):
        if self.samples == 0:
            self.get_logger().warn('no synchronized samples yet '
                                   '(is depth_obstacle_node running?)')
            return
        n = self.samples
        pct_frames = 100.0 * self.depth_only_frames / n
        near = (f'{np.median(self.nearest_depth_only):.1f} m'
                if self.nearest_depth_only else 'n/a')
        self.get_logger().info(
            f'[{n} samples] mean cells/frame -- '
            f'AGREE {self.agree_total/n:.0f} | '
            f'DEPTH-ONLY {self.depth_only_total/n:.0f} | '
            f'RGB-ONLY {self.rgb_only_total/n:.0f} || '
            f'frames with any depth-only: {pct_frames:.0f}% | '
            f'median nearest depth-only: {near}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report-period', type=float, default=10.0)
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = DepthRgbCrosscheck(args.report_period)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
