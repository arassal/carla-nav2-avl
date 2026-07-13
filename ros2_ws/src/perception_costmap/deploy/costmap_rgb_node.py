#!/usr/bin/env python3
"""Colorized costmap -> RGB PointCloud2 for RViz.

RViz's Map display only ships fixed colour schemes, none of which say what we
want, so we publish the grid as an RGB PointCloud2 (Style: Boxes, one cell
each) and own the palette:

    black   unknown -- never observed
    green   free / low cost      (go)
    orange  medium cost
    red     high cost / lethal   (bad)

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

# cost -> colour ramp. green = go, orange = caution, red = bad. Anchors are
# lerped, so a graded cost field reads as a continuous fade rather than bands.
RAMP = np.array([
    (0,    30, 200,  70),    # free        -> green
    (25,   90, 205,  55),    # low         -> green
    (50,  255, 170,  30),    # medium      -> orange
    (75,  255, 110,  25),    # high        -> deep orange
    (97,  240,  55,  40),    # off-road    -> red
    (100, 255,   0,   0),    # lethal      -> bright red
], dtype=np.float32)

UNKNOWN_RGB = (12, 12, 16)   # near-black


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
        self.lut = build_lut()
        self.known = None       # bool array, from /perception/known

        self.create_subscription(OccupancyGrid, '/perception/known',
                                 self._known_cb, 1)
        self.create_subscription(OccupancyGrid, '/perception/costmap',
                                 self._cost_cb, 1)
        self.pub = self.create_publisher(PointCloud2, '/viz/costmap_rgb', 1)
        self.get_logger().info(
            'costmap_rgb up -> /viz/costmap_rgb (waiting for /perception/known)')

    def _known_cb(self, msg):
        self.known = (np.array(msg.data, np.int8)
                      .reshape(msg.info.height, msg.info.width) > 0)

    def _cost_cb(self, msg):
        if self.known is None:
            return      # without the authoritative mask we cannot say what is
                        # unknown, and guessing is exactly the bug we fixed
        h, w = msg.info.height, msg.info.width
        if self.known.shape != (h, w):
            return

        grid = np.array(msg.data, np.int16).reshape(h, w)
        rgb = self.lut[np.clip(grid, 0, 100).astype(np.uint8)]
        # Blind cells now carry an infilled GUESS (see occupancy.infill_unknown),
        # so render them in the guessed colour but dimmed to ~40% -- visibly
        # "we think this, but nothing has seen it". grid < 0 (true ROS unknown,
        # not emitted by this stack) stays black.
        guessed = ~self.known
        rgb[guessed] = (rgb[guessed].astype(np.float32) * 0.4).astype(np.uint8)
        rgb[grid < 0] = UNKNOWN_RGB

        res = msg.info.resolution
        xs = msg.info.origin.position.x + (np.arange(w) + 0.5) * res
        ys = msg.info.origin.position.y + (np.arange(h) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)

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
