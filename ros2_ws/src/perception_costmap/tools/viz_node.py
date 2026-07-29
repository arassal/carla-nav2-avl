#!/usr/bin/env python3
"""
Lightweight live-viz node for the dinosaur dashboard.

Subscribes to the 3 ZED RGB streams and /perception/costmap, publishes two
render topics for web_video_server (no NN inference here — the costmap
already encodes what the learned models saw, so this stays cheap):

  /viz/fused_bev        3-camera BEV composite, forward = up
  /viz/costmap_render   colorized live costmap, same orientation

Run from the package root (same as the other tools), with the ZED wrappers,
sensors.launch.py and the costmap node up:

    cd ros2_ws/src/perception_costmap
    PYTHONPATH=.:$PYTHONPATH python3 tools/viz_node.py

Serve to a browser with web_video_server (see deploy/live_dashboard.html):

    ros2 run web_video_server web_video_server --ros-args -p port:=8080

Camera mounts below are dinosaur's (from avros.urdf.xacro) -- edit CAMS for
a different vehicle.

Render rules (2026-07-02, per operator feedback):
  - REP-103 top-down orientation: forward = up, car's LEFT = image left
    (the first version was mirrored — transpose needs a flip on BOTH axes)
  - each camera's wedge is range-clipped (RANGE_M) so near-horizon pixels
    don't smear to the grid edges
  - the rear sector (no rear camera, car never reverses) is hard-masked
    dark and labeled — never painted by any camera
  - overlaps are feather-blended (distance-transform weights)
  - display cropped to the useful envelope (X_DISP_MAX forward) so content
    fills the frame at true aspect; range rings + car footprint for the
    top-down read
"""
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

from perception_costmap.bev import homography_from_extrinsics, warp_to_bev, bev_known_mask
from perception_costmap.occupancy import GridSpec

GRID = GridSpec(x_min=-4.0, x_max=16.0, y_min=-10.0, y_max=10.0, resolution=0.1)
RANGE_M = 12.0          # max trustworthy IPM distance from each camera
REAR_DEG = 135.0        # |bearing from ego| beyond this = rear blind sector
STALE_SEC = 1.5         # drop a camera wedge / flag the costmap after this
X_DISP_MAX = 13.0       # crop display here: content ends at RANGE_M anyway
PX_PER_M = 24           # display scale -> 480px wide for the 20 m y-span

# car footprint from avros.urdf.xacro properties (verified on the car
# 2026-07-02: main_length=0.7430, main_width=0.6795; the width matching the
# front camera's cam_x=0.6795 is a coincidence)
CAR_LEN, CAR_WID = 0.743, 0.6795

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


def grid_world_coords():
    """Cell-center world coords X[i,j], Y[i,j] for the raw (row=y, col=x)
    BEV array orientation used by warp_to_bev."""
    gh = int(round((GRID.y_max - GRID.y_min) / GRID.resolution))
    gw = int(round((GRID.x_max - GRID.x_min) / GRID.resolution))
    xs = GRID.x_min + (np.arange(gw) + 0.5) * GRID.resolution
    ys = GRID.y_min + (np.arange(gh) + 0.5) * GRID.resolution
    X, Y = np.meshgrid(xs, ys)
    return X, Y, gh, gw


class VizNode(Node):
    def __init__(self):
        super().__init__('perception_viz')
        self.bridge = CvBridge()
        self.imgs = {}
        self.img_t = {}      # per-camera receive time (monotonic)
        self.H = {}
        self.weight = {}     # feathered blend weight per camera (float32)
        self.grid_msg = None
        self.grid_t = 0.0

        self.X, self.Y, self.gh, self.gw = grid_world_coords()
        bearing = np.degrees(np.arctan2(self.Y, self.X))
        self.rear_mask = np.abs(bearing) > REAR_DEG      # blind sector

        # display geometry: crop x <= X_DISP_MAX, render at PX_PER_M
        self.crop_rows = int(round((GRID.x_max - X_DISP_MAX) / GRID.resolution))
        disp_w_m = GRID.y_max - GRID.y_min               # 20 m
        disp_h_m = X_DISP_MAX - GRID.x_min               # 17 m
        self.disp_w = int(disp_w_m * PX_PER_M)           # 480
        self.disp_h = int(disp_h_m * PX_PER_M)           # 408
        # ego position in display px (world origin)
        self.ego_px = (int((GRID.y_max - 0.0) * PX_PER_M),        # col: +y = left
                       int((X_DISP_MAX - 0.0) * PX_PER_M))        # row: +x = up

        for name, c in CAMS.items():
            self.create_subscription(Image, c['topic'],
                                     self._img_cb(name), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, c['info'],
                                     self._info_cb(name), qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, '/perception/costmap',
                                 self._cost_cb, 1)

        self.bev_pub = self.create_publisher(Image, '/viz/fused_bev', 1)
        self.cost_pub = self.create_publisher(Image, '/viz/costmap_render', 1)
        self.create_timer(0.2, self._tick)          # 5 Hz render
        self.get_logger().info('perception_viz up: /viz/fused_bev /viz/costmap_render')

    def _img_cb(self, name):
        def cb(msg):
            self.imgs[name] = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.img_t[name] = time.monotonic()
        return cb

    def _info_cb(self, name):
        def cb(msg):
            if name not in self.H and name in self.imgs:
                K = np.array(msg.k, float).reshape(3, 3)
                c = CAMS[name]
                H = homography_from_extrinsics(K, c['xyz'], c['pitch'], c['yaw'], GRID)
                known = bev_known_mask(H, self.imgs[name].shape, GRID).astype(bool)
                dist = np.hypot(self.X - c['xyz'][0], self.Y - c['xyz'][1])
                mask = known & (dist <= RANGE_M) & ~self.rear_mask
                w = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
                np.clip(w, 0, 15, out=w)             # cap feather width 1.5 m
                self.H[name] = H
                self.weight[name] = w.astype(np.float32)
                self.get_logger().info(f'homography ready: {name}')
        return cb

    def _cost_cb(self, msg):
        self.grid_msg = msg
        self.grid_t = time.monotonic()

    # raw (row=y, col=x) -> top-down display: forward up, car-left = image
    # left. transpose then flip BOTH axes (single-axis flip mirrors it).
    def _to_display(self, raw, interp):
        up = cv2.flip(cv2.transpose(raw), -1)
        up = up[self.crop_rows:, :]                  # crop beyond X_DISP_MAX
        return cv2.resize(up, (self.disp_w, self.disp_h), interpolation=interp)

    def _tick(self):
        gh, gw = self.gh, self.gw

        now = time.monotonic()
        if self.H:
            acc = np.zeros((gh, gw, 3), np.float32)
            wsum = np.zeros((gh, gw), np.float32)
            stale = []
            for name in CAMS:
                if name not in self.H or name not in self.imgs:
                    continue
                if now - self.img_t.get(name, 0.0) > STALE_SEC:
                    stale.append(name)       # dead feed: dark wedge, not a
                    continue                 # frozen last frame
                bev_rgb = warp_to_bev(self.imgs[name], self.H[name], GRID,
                                      interp=cv2.INTER_LINEAR)
                w = self.weight[name]
                acc += bev_rgb.astype(np.float32) * w[..., None]
                wsum += w
            seen = wsum > 0
            fused = np.zeros((gh, gw, 3), np.uint8)
            fused[seen] = (acc[seen] / wsum[seen, None]).astype(np.uint8)
            fused[~seen] = (20, 20, 20)
            fused[self.rear_mask] = (10, 10, 14)
            img = self._to_display(fused, cv2.INTER_LINEAR)
            self._draw_overlay(img)
            for k, name in enumerate(stale):
                cv2.putText(img, '%s STALE' % name.upper(), (8, 18 + 16 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 230), 1,
                            cv2.LINE_AA)
            self.bev_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

        if self.grid_msg is not None:
            m = self.grid_msg
            g = np.array(m.data, dtype=np.int8).reshape(m.info.height, m.info.width)
            vis = np.zeros((*g.shape, 3), np.uint8)
            vis[g == 0]   = (190, 190, 190)
            vis[g == 100] = (60, 60, 230)
            vis[g == -1]  = (45, 45, 45)
            mid = (g > 0) & (g < 100)        # inflation/gradient costs
            if mid.any():
                inten = (180 - g[mid]).astype(np.uint8)
                vis[mid] = np.stack((inten, inten, inten), axis=-1)
            if vis.shape[:2] == (gh, gw):
                vis[self.rear_mask] = (10, 10, 14)
            img = self._to_display(vis, cv2.INTER_NEAREST)   # keep cells crisp
            self._draw_overlay(img)
            age = now - self.grid_t
            if age > STALE_SEC:
                img //= 2
                cv2.putText(img, 'COSTMAP STALE %.0fs' % age, (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 230), 1,
                            cv2.LINE_AA)
            self.cost_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def _draw_overlay(self, img):
        ex, ey = self.ego_px
        ring = (78, 78, 84)
        # range rings only over the covered sector (±REAR_DEG around forward).
        # cv2 angles: 0 deg = image +x, clockwise. forward(up) = -90 deg.
        for r_m in range(2, int(RANGE_M) + 1, 2):
            r = r_m * PX_PER_M
            cv2.ellipse(img, (ex, ey), (r, r), 0, -90 - REAR_DEG,
                        -90 + REAR_DEG, ring, 1, cv2.LINE_AA)
            if r_m % 4 == 0 and ey - r > 12:
                cv2.putText(img, f'{r_m}m', (ex + 4, ey - r + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (120, 120, 126),
                            1, cv2.LINE_AA)
        # car footprint (URDF chassis dims) + heading tick
        hw = int(CAR_WID / 2 * PX_PER_M)
        hl = int(CAR_LEN / 2 * PX_PER_M)
        cv2.rectangle(img, (ex - hw, ey - hl), (ex + hw, ey + hl),
                      (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (ex, ey - hl), (ex, ey - hl - 8), (255, 255, 255), 2,
                 cv2.LINE_AA)
        cv2.putText(img, 'NO REAR CAM', (ex - 44, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 120), 1,
                    cv2.LINE_AA)


def main():
    rclpy.init()
    node = VizNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
