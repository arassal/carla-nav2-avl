#!/usr/bin/env python3
"""Colorized costmap -> RGB PointCloud2 for RViz.

RViz's Map display only offers fixed colour schemes ("map"/"costmap"/"raw"),
none of which say what we want. So we publish the grid as an RGB PointCloud2
instead (Style: Boxes, size = one cell) and own the palette:

    black   unknown  -- never observed
    green   free / low cost      (go)
    orange  medium cost
    red     high cost / lethal   (bad)

WHY UNKNOWN IS COMPUTED, NOT READ:
perception_dinosaur.yaml sets `unknown_cost: 25`, so unobserved cells are
published in the OccupancyGrid as the *value 25* -- numerically identical to
a genuine medium-low cost cell. The grid alone therefore cannot tell you what
is unknown. We recover the truth geometrically with the same coverage test
the production viz node uses (perception_costmap.bev): a cell is KNOWN only
if it falls inside some camera's ground homography, within RANGE_M of that
camera, and outside the rear blind sector. Everything else is unknown -> black.
That is why the rear wedge (no rear camera exists) renders black.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from cv_bridge import CvBridge

from perception_costmap.bev import homography_from_extrinsics, bev_known_mask
from perception_costmap.occupancy import GridSpec

GRID = GridSpec(x_min=-4.0, x_max=16.0, y_min=-10.0, y_max=10.0, resolution=0.1)
RANGE_M = 12.0          # max trustworthy IPM distance from each camera
REAR_DEG = 135.0        # |bearing from ego| beyond this = rear blind sector

# Same extrinsics as tools/viz_node.py (measured on the car 2026-07-09).
CAMS = {
    'front': dict(topic='/zed_front/zed_node/rgb/color/rect/image',
                  info='/zed_front/zed_node/rgb/color/rect/camera_info',
                  xyz=(0.6795, 0.0, 0.4476), pitch=15.0, yaw=0.0),
    'left':  dict(topic='/zed_left/zed_node/rgb/color/rect/image',
                  info='/zed_left/zed_node/rgb/color/rect/camera_info',
                  xyz=(0.098, 0.286, 0.6126), pitch=18.0, yaw=80.0),
    'right': dict(topic='/zed_right/zed_node/rgb/color/rect/image',
                  info='/zed_right/zed_node/rgb/color/rect/camera_info',
                  xyz=(0.098, -0.286, 0.6126), pitch=18.0, yaw=-80.0),
}

# cost -> colour ramp. green = go, orange = caution, red = bad.
# Anchors are (cost, R, G, B); we lerp between them, so the ramp reads
# continuously instead of banding into 3 flat buckets.
RAMP = np.array([
    (0,    30, 200,  70),    # free            -> green
    (25,   90, 205,  55),    # low             -> green
    (50,  255, 170,  30),    # medium          -> orange
    (75,  255, 110,  25),    # high            -> deep orange
    (99,  235,  40,  40),    # near-lethal     -> red
    (100, 255,   0,   0),    # lethal          -> bright red
], dtype=np.float32)

UNKNOWN_RGB = (12, 12, 16)   # near-black


def build_lut():
    """256-entry RGB LUT indexed by cost 0..100 (rest unused)."""
    lut = np.zeros((256, 3), np.uint8)
    costs = RAMP[:, 0]
    for v in range(101):
        r = np.interp(v, costs, RAMP[:, 1])
        g = np.interp(v, costs, RAMP[:, 2])
        b = np.interp(v, costs, RAMP[:, 3])
        lut[v] = (int(r), int(g), int(b))
    return lut


class CostmapRGB(Node):
    def __init__(self):
        super().__init__('costmap_rgb')
        self.bridge = CvBridge()
        self.lut = build_lut()
        self.imgs = {}
        self.coverage = {}          # per-camera known mask (bool)

        gh = int(round((GRID.y_max - GRID.y_min) / GRID.resolution))
        gw = int(round((GRID.x_max - GRID.x_min) / GRID.resolution))
        xs = GRID.x_min + (np.arange(gw) + 0.5) * GRID.resolution
        ys = GRID.y_min + (np.arange(gh) + 0.5) * GRID.resolution
        self.X, self.Y = np.meshgrid(xs, ys)
        self.gh, self.gw = gh, gw

        bearing = np.degrees(np.arctan2(self.Y, self.X))
        self.rear_mask = np.abs(bearing) > REAR_DEG

        for name, c in CAMS.items():
            self.create_subscription(Image, c['topic'],
                                     self._img_cb(name), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, c['info'],
                                     self._info_cb(name), qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/perception/costmap',
                                 self._cost_cb, 1)
        self.pub = self.create_publisher(PointCloud2, '/viz/costmap_rgb', 1)
        self.get_logger().info('costmap_rgb up -> /viz/costmap_rgb (waiting for camera_info)')

    def _img_cb(self, name):
        def cb(msg):
            self.imgs[name] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        return cb

    def _info_cb(self, name):
        def cb(msg):
            if name in self.coverage or name not in self.imgs:
                return
            K = np.array(msg.k, float).reshape(3, 3)
            c = CAMS[name]
            H = homography_from_extrinsics(K, c['xyz'], c['pitch'], c['yaw'], GRID)
            known = bev_known_mask(H, self.imgs[name].shape, GRID).astype(bool)
            dist = np.hypot(self.X - c['xyz'][0], self.Y - c['xyz'][1])
            self.coverage[name] = known & (dist <= RANGE_M) & ~self.rear_mask
            self.get_logger().info(f'coverage ready: {name} ({int(self.coverage[name].sum())} cells)')
        return cb

    def known_mask(self):
        if not self.coverage:
            return None
        m = np.zeros((self.gh, self.gw), bool)
        for c in self.coverage.values():
            m |= c
        return m

    def _cost_cb(self, msg):
        known = self.known_mask()
        if known is None:
            return      # no homographies yet; nothing trustworthy to say

        grid = np.array(msg.data, np.int16).reshape(msg.info.height, msg.info.width)

        cost = np.clip(grid, 0, 100).astype(np.uint8)
        rgb = self.lut[cost]                       # (h,w,3)
        # -1 would be ROS's real "unknown"; this stack never emits it (see
        # module docstring) but honour it anyway if that ever changes.
        unobserved = (~known) | (grid < 0)
        rgb[unobserved] = UNKNOWN_RGB

        self.pub.publish(self._cloud(msg.header, rgb))

    def _cloud(self, header, rgb):
        res = GRID.resolution
        x = self.X.ravel().astype(np.float32)
        y = self.Y.ravel().astype(np.float32)
        z = np.zeros_like(x)

        r = rgb[..., 0].ravel().astype(np.uint32)
        g = rgb[..., 1].ravel().astype(np.uint32)
        b = rgb[..., 2].ravel().astype(np.uint32)
        packed = (r << 16) | (g << 8) | b
        rgb_f = packed.view(np.float32) if packed.dtype == np.float32 else \
            packed.astype(np.uint32).view(np.float32)

        pts = np.zeros((x.size, 4), np.float32)
        pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3] = x, y, z, rgb_f

        msg = PointCloud2()
        msg.header = Header(stamp=header.stamp, frame_id=header.frame_id)
        msg.height = 1
        msg.width = x.size
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.data = pts.tobytes()
        return msg


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
