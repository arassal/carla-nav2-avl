#!/usr/bin/env python3
"""Colorized costmap -> RGB PointCloud2 for RViz.

RViz's Map display only ships fixed colour schemes, none of which say what we
want, so we publish the grid as an RGB PointCloud2 (Style: Boxes, one cell
each) and own the palette:

    charcoal  unknown -- never observed
    green     free / low cost
    yellow    medium cost
    red       high cost / lethal

WHY WE SUBSCRIBE TO /perception/known:
perception_dinosaur.yaml sets `unknown_cost: 25`, so unobserved cells are
published in the OccupancyGrid as the literal value 25 -- indistinguishable
from a genuine medium-low cost. The grid alone therefore cannot tell you what
is unknown, and colouring by value paints the blind region as low-cost "go"
ground.

An earlier version of this node recomputed the coverage mask itself from the
camera homographies. That is a second implementation of the same idea, and it
drifted: it applied a 12 m range clip and a rear cutoff that costmap_node does
not, so cells the costmap considered UNKNOWN were shown as observed and
coloured green. The costmap node now publishes its own `known` mask and we use
that -- one source of truth.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

# Green -> yellow -> red gives the operator an immediate go/caution/stop view.
# Anchors are interpolated so the cost gradient remains visible between bands.
RAMP = np.array([
    (0,    30, 200,  70),    # free        -> green
    (25,   90, 205,  55),    # low         -> yellow-green
    (50,  255, 170,  30),    # medium      -> amber
    (75,  255, 110,  25),    # high        -> orange
    (96,  240,  55,  40),    # near lethal -> red
    (97,  245,  40,  35),    # off-road    -> stronger red
    (100, 255,   0,   0),    # lethal      -> red
], dtype=np.float32)

UNKNOWN_DARK_RGB = (31, 35, 41)
UNKNOWN_LIGHT_RGB = (42, 47, 54)
ROS_UNKNOWN_RGB = (12, 14, 18)


def build_lut():
    lut = np.zeros((256, 3), np.uint8)
    costs = RAMP[:, 0]
    for v in range(101):
        lut[v] = (int(np.interp(v, costs, RAMP[:, 1])),
                  int(np.interp(v, costs, RAMP[:, 2])),
                  int(np.interp(v, costs, RAMP[:, 3])))
    return lut


class CostmapRGB(Node):
    def __init__(self):
        super().__init__('costmap_rgb')
        self.declare_parameter('publish_rate', 2.0)
        self.lut = build_lut()
        self.known = None       # bool array, from /perception/known
        self.latest_cost = None
        self.last_stamp = None
        self.geometry_key = None
        self.geometry = None

        self.create_subscription(OccupancyGrid, '/perception/known',
                                 self._known_cb, 1)
        self.create_subscription(OccupancyGrid, '/perception/costmap',
                                 self._cost_cb, 1)
        self.pub = self.create_publisher(PointCloud2, '/viz/costmap_rgb', 1)
        publish_rate = float(self.get_parameter('publish_rate').value)
        if publish_rate <= 0.0:
            raise ValueError('publish_rate must be positive')
        self.create_timer(1.0 / publish_rate, self._publish_latest)
        self.get_logger().info(
            f'costmap_rgb up -> /viz/costmap_rgb at {publish_rate:.1f} Hz '
            '(waiting for /perception/known)')

    def _known_cb(self, msg):
        self.known = (np.array(msg.data, np.int8)
                      .reshape(msg.info.height, msg.info.width) > 0)

    def _cost_cb(self, msg):
        self.latest_cost = msg

    def _publish_latest(self):
        msg = self.latest_cost
        if msg is None or self.known is None:
            return      # without the authoritative mask we cannot say what is
                        # unknown, and guessing is exactly the bug we fixed
        h, w = msg.info.height, msg.info.width
        if self.known.shape != (h, w):
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == self.last_stamp:
            return
        self.last_stamp = stamp

        grid = np.array(msg.data, np.int16).reshape(h, w)
        rgb = self.lut[np.clip(grid, 0, 100).astype(np.uint8)]
        # Navigation may assign a numeric prior to unseen cells, but the
        # operator display must not make guessed space look observed. Render a
        # coarse neutral checker pattern that remains legible when zoomed out.
        guessed = ~self.known
        rows, cols = np.indices((h, w))
        checker = ((rows // 5 + cols // 5) % 2).astype(bool)
        rgb[guessed & ~checker] = UNKNOWN_DARK_RGB
        rgb[guessed & checker] = UNKNOWN_LIGHT_RGB
        rgb[grid < 0] = ROS_UNKNOWN_RGB

        key = (h, w, msg.info.resolution,
               msg.info.origin.position.x, msg.info.origin.position.y)
        if key != self.geometry_key:
            res = msg.info.resolution
            xs = msg.info.origin.position.x + (np.arange(w) + 0.5) * res
            ys = msg.info.origin.position.y + (np.arange(h) + 0.5) * res
            self.geometry = np.meshgrid(xs, ys)
            self.geometry_key = key
        X, Y = self.geometry

        r = rgb[..., 0].ravel().astype(np.uint32)
        g = rgb[..., 1].ravel().astype(np.uint32)
        b = rgb[..., 2].ravel().astype(np.uint32)
        packed = ((r << 16) | (g << 8) | b).astype(np.uint32).view(np.float32)

        pts = np.zeros((X.size, 4), np.float32)
        pts[:, 0] = X.ravel()
        pts[:, 1] = Y.ravel()
        pts[:, 2] = 0.0
        pts[:, 3] = packed

        out = PointCloud2()
        out.header = Header(stamp=msg.header.stamp, frame_id=msg.header.frame_id)
        out.height = 1
        out.width = X.size
        out.is_dense = True
        out.is_bigendian = False
        out.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out.point_step = 16
        out.row_step = out.point_step * out.width
        out.data = pts.tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = CostmapRGB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
