"""
ZED X Camera → ROS2 Bridge Node

Subscribes to three ZED X cameras (front, left, right) and produces
the same /scan and /ped_scan LaserScan topics that costmap_node.py
consumes — so the costmap layer is completely unchanged.

Pipeline per camera:
  /zed_front|left|right/depth/depth_registered  (32-bit float depth, metres)
  /zed_front|left|right/rgb/image_rect_color    (BGR8, for YOLO detection)
  /zed_front|left|right/rgb/camera_info         (intrinsics K matrix)

  → Project each depth pixel to a 3D point (camera frame)
  → Transform to base_link frame via TF
  → Run YOLO on RGB to get pedestrian bounding boxes
  → Split points into pedestrian / non-pedestrian
  → Bin by horizontal angle → publish as LaserScan

Requires:
  pip3 install ultralytics opencv-python
  ZED SDK + zed-ros2-wrapper running (provides the camera topics above)
"""

import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from cv_bridge import CvBridge

import tf2_ros

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Quaternion

# ── tunables ──────────────────────────────────────────────────────────────────

# Topic prefixes for each ZED X camera (match your zed-ros2-wrapper config)
CAMERAS = {
    'front': '/zed_front',
    'left':  '/zed_left',
    'right': '/zed_right',
}

LIDAR_RAYS    = 720        # angular bins in the output LaserScan
RANGE_MIN     = 0.3        # metres — ignore very close returns
RANGE_MAX     = 20.0       # metres — ZED X reliable depth range
HEIGHT_MIN    = -0.5       # metres in base_link z — ignore ground plane
HEIGHT_MAX    =  2.5       # metres in base_link z — ignore sky / treetops

# YOLO
PED_CONF      = 0.40
YOLO_MODEL    = 'yolov8n.pt'   # nano = fast; swap for yolov8m for better recall
YOLO_PED_CLS  = 0              # COCO class 0 = person

PUB_HZ        = 10.0

# ── helpers ───────────────────────────────────────────────────────────────────

def _q_to_mat(x, y, z, w):
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float64)


class CameraBuffer:
    """Thread-safe store for the latest data from one ZED X camera."""
    def __init__(self):
        self.lock      = threading.Lock()
        self.depth     = None   # np.ndarray float32 HxW, metres
        self.rgb       = None   # np.ndarray uint8 HxWx3, BGR
        self.K         = None   # 3x3 camera intrinsics
        self.frame_id  = None   # optical frame name from the depth header


# ── node ──────────────────────────────────────────────────────────────────────

class ZedBridgeNode(Node):
    def __init__(self):
        super().__init__('zed_bridge')
        self._cbg    = ReentrantCallbackGroup()
        self._bridge = CvBridge()

        # Load YOLO once at startup
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(YOLO_MODEL)
            self.get_logger().info(f'YOLO loaded: {YOLO_MODEL}')
        except Exception as e:
            self._yolo = None
            self.get_logger().warn(
                f'YOLO unavailable ({e}) — pedestrian split disabled, '
                'all points go to /scan')

        # TF
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # Per-camera buffers
        self._bufs = {name: CameraBuffer() for name in CAMERAS}

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        for name, prefix in CAMERAS.items():
            buf = self._bufs[name]
            self.create_subscription(
                Image, f'{prefix}/depth/depth_registered',
                lambda m, b=buf: self._on_depth(m, b),
                sensor_qos, callback_group=self._cbg)
            self.create_subscription(
                Image, f'{prefix}/rgb/image_rect_color',
                lambda m, b=buf: self._on_rgb(m, b),
                sensor_qos, callback_group=self._cbg)
            self.create_subscription(
                CameraInfo, f'{prefix}/rgb/camera_info',
                lambda m, b=buf: self._on_info(m, b),
                sensor_qos, callback_group=self._cbg)

        self._pub_scan     = self.create_publisher(LaserScan, '/scan',     10)
        self._pub_ped_scan = self.create_publisher(LaserScan, '/ped_scan', 10)

        self.create_timer(1.0 / PUB_HZ, self._publish, callback_group=self._cbg)

        self.get_logger().info(
            f'ZED bridge ready — cameras: {list(CAMERAS.keys())}, '
            f'range {RANGE_MIN}–{RANGE_MAX} m, {LIDAR_RAYS} bins')

    # ── camera callbacks ──────────────────────────────────────────────────────

    def _on_depth(self, msg: Image, buf: CameraBuffer):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with buf.lock:
                buf.depth    = depth
                buf.frame_id = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'depth decode: {e}', throttle_duration_sec=5)

    def _on_rgb(self, msg: Image, buf: CameraBuffer):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with buf.lock:
                buf.rgb = rgb
        except Exception as e:
            self.get_logger().warn(f'rgb decode: {e}', throttle_duration_sec=5)

    def _on_info(self, msg: CameraInfo, buf: CameraBuffer):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        with buf.lock:
            buf.K = K

    # ── pedestrian mask ───────────────────────────────────────────────────────

    def _ped_mask(self, rgb, H: int, W: int) -> np.ndarray:
        """Boolean (H x W) mask — True inside detected pedestrian bboxes."""
        mask = np.zeros((H, W), dtype=bool)
        if self._yolo is None or rgb is None:
            return mask
        try:
            results = self._yolo(rgb, classes=[YOLO_PED_CLS],
                                 conf=PED_CONF, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W - 1, x2), min(H - 1, y2)
                    mask[y1:y2, x1:x2] = True
        except Exception as e:
            self.get_logger().warn(f'YOLO: {e}', throttle_duration_sec=5)
        return mask

    # ── depth projection ──────────────────────────────────────────────────────

    def _project(self, depth: np.ndarray, K: np.ndarray, frame_id: str):
        """
        Back-project depth image to 3D points in base_link frame.
        Returns (pts_bl Nx3, row_indices, col_indices) or None.
        """
        try:
            tf = self._tf_buf.lookup_transform(
                'base_link', frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0))
        except Exception:
            return None

        H, W   = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        us = np.arange(W, dtype=np.float32)
        vs = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)

        z     = depth.astype(np.float32)
        valid = np.isfinite(z) & (z > RANGE_MIN) & (z < RANGE_MAX)

        zv = z[valid]
        uv = uu[valid]
        vv_ = vv[valid]

        # Optical-frame 3D (x=right, y=down, z=forward for ZED optical frame)
        xc = (uv - cx) / fx * zv
        yc = (vv_ - cy) / fy * zv
        zc = zv

        t   = tf.transform.translation
        q   = tf.transform.rotation
        rot = _q_to_mat(q.x, q.y, q.z, q.w)

        pts_cam = np.stack([xc, yc, zc], axis=1)
        pts_bl  = (rot @ pts_cam.T).T + np.array([t.x, t.y, t.z])

        # Height filter
        h_ok   = (pts_bl[:, 2] > HEIGHT_MIN) & (pts_bl[:, 2] < HEIGHT_MAX)
        pts_bl = pts_bl[h_ok]

        valid_flat = np.flatnonzero(valid)[h_ok]
        rows = valid_flat // W
        cols = valid_flat  % W

        return pts_bl, rows, cols

    # ── publish ───────────────────────────────────────────────────────────────

    def _publish(self):
        obs_ranges = [float('inf')] * LIDAR_RAYS
        ped_ranges = [float('inf')] * LIDAR_RAYS

        for name, buf in self._bufs.items():
            with buf.lock:
                depth    = buf.depth
                rgb      = buf.rgb
                K        = buf.K
                frame_id = buf.frame_id

            if depth is None or K is None or frame_id is None:
                continue

            H, W   = depth.shape
            result = self._project(depth, K, frame_id)
            if result is None:
                continue
            pts_bl, rows, cols = result

            ped_px = self._ped_mask(rgb, H, W)
            is_ped = ped_px[rows, cols]

            x = pts_bl[:, 0]
            y = pts_bl[:, 1]
            d = np.hypot(x, y)

            for i in range(len(pts_bl)):
                di = float(d[i])
                if di < RANGE_MIN or di > RANGE_MAX:
                    continue
                angle = math.atan2(float(y[i]), float(x[i]))
                idx   = int((angle + math.pi) / (2 * math.pi) * LIDAR_RAYS) % LIDAR_RAYS
                if is_ped[i]:
                    if di < ped_ranges[idx]:
                        ped_ranges[idx] = di
                else:
                    if di < obs_ranges[idx]:
                        obs_ranges[idx] = di

        now = self.get_clock().now().to_msg()
        self._pub_scan.publish(    self._make_scan(obs_ranges, now))
        self._pub_ped_scan.publish(self._make_scan(ped_ranges, now))

    def _make_scan(self, ranges, stamp):
        scan = LaserScan()
        scan.header.stamp    = stamp
        scan.header.frame_id = 'base_link'
        scan.angle_min       = -math.pi
        scan.angle_max       =  math.pi
        scan.angle_increment = 2 * math.pi / LIDAR_RAYS
        scan.time_increment  = 0.0
        scan.scan_time       = 1.0 / PUB_HZ
        scan.range_min       = float(RANGE_MIN)
        scan.range_max       = float(RANGE_MAX)
        scan.ranges          = ranges
        return scan


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = ZedBridgeNode()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
