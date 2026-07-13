"""Three-ZED perception node deriving six costmap layers from RGB/depth only."""

import math
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .core import GridSpec, TemporalVoxelGrid
from .perception import (CameraSample, CentroidTracker, PersistenceSeparator,
                         ThreeCameraSynchronizer, depth_to_base_points,
                         derive_layers, stamp_seconds)


class CameraBuffer:
    def __init__(self):
        self.depth = self.rgb = self.k = None
        self.depth_stamp = self.rgb_stamp = None


class ThreeZedPerceptionNode(Node):
    def __init__(self):
        super().__init__('three_zed_perception')
        self.declare_parameter('camera_names', ['zed_front', 'zed_left', 'zed_right'])
        self.declare_parameter('max_camera_skew_s', 0.050)
        self.declare_parameter('max_rgb_depth_skew_s', 0.035)
        self.declare_parameter('stale_camera_s', 0.5)
        self.declare_parameter('processing_frequency', 10.0)
        self.declare_parameter('depth_stride', 4)
        self.declare_parameter('min_depth_m', 0.5)
        self.declare_parameter('max_depth_m', 25.0)
        self.declare_parameter('width_m', 60.0)
        self.declare_parameter('height_m', 60.0)
        self.declare_parameter('resolution', 0.20)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('inflation_radius_m', 2.5)
        self.declare_parameter('voxel_persistence_s', 2.0)
        # Provisional sketch-derived mount values. Override after calibration.
        self.declare_parameter('mounts.front.translation', [0.67945, 0.0, -0.10795])
        self.declare_parameter('mounts.front.yaw', 0.0)
        self.declare_parameter('mounts.left.translation', [0.098425, -0.28575, 0.05715])
        self.declare_parameter('mounts.left.yaw', math.pi / 2)
        self.declare_parameter('mounts.right.translation', [0.098425, 0.28575, 0.05715])
        self.declare_parameter('mounts.right.yaw', -math.pi / 2)

        camera_names = list(self.get_parameter('camera_names').value)
        if len(camera_names) != 3:
            raise ValueError('exactly three camera_names are required')
        self._logical = dict(zip(('front', 'left', 'right'), camera_names))
        self._bridge = CvBridge()
        self._buffers = {logical: CameraBuffer() for logical in self._logical}
        self._lock = threading.Lock()
        self._sync = ThreeCameraSynchronizer(max_skew_s=float(
            self.get_parameter('max_camera_skew_s').value))
        self._spec = GridSpec(float(self.get_parameter('width_m').value),
                              float(self.get_parameter('height_m').value),
                              float(self.get_parameter('resolution').value))
        self._separator = PersistenceSeparator(self._spec.shape)
        self._tracker = CentroidTracker()
        self._voxel_grid = TemporalVoxelGrid(
            self._spec, persistence_s=float(self.get_parameter('voxel_persistence_s').value))
        self._last_input_wall = {name: 0.0 for name in self._logical}
        self._publishers = {name: self.create_publisher(
            OccupancyGrid, f'/seven_layer_costmap/layers/{name}', 1)
            for name in ('lanelet', 'static_obstacle', 'spatio_temporal_voxel',
                         'prediction', 'inflation', 'traffic_regulation')}
        self._status_pub = self.create_publisher(String, '/seven_layer_costmap/perception_status', 10)
        for logical, camera in self._logical.items():
            prefix = f'/{camera}/{camera}_node'
            self.create_subscription(Image, prefix + '/depth/depth_registered',
                                     lambda m, n=logical: self._depth(n, m), qos_profile_sensor_data)
            self.create_subscription(Image, prefix + '/left/color/rect/image',
                                     lambda m, n=logical: self._rgb(n, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, prefix + '/left/camera_info',
                                     lambda m, n=logical: self._info(n, m), qos_profile_sensor_data)
        hz = float(self.get_parameter('processing_frequency').value)
        self.create_timer(1.0 / hz, self._process)

    def _depth(self, name, msg):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, '32FC1')
            with self._lock:
                self._buffers[name].depth = depth
                self._buffers[name].depth_stamp = stamp_seconds(msg.header.stamp)
                self._last_input_wall[name] = time.monotonic()
        except Exception as error:
            self.get_logger().warn(f'{name} depth decode: {error}', throttle_duration_sec=5.0)

    def _rgb(self, name, msg):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._lock:
                self._buffers[name].rgb = rgb
                self._buffers[name].rgb_stamp = stamp_seconds(msg.header.stamp)
        except Exception as error:
            self.get_logger().warn(f'{name} RGB decode: {error}', throttle_duration_sec=5.0)

    def _info(self, name, msg):
        with self._lock:
            self._buffers[name].k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)

    def _collect(self):
        allowed = float(self.get_parameter('max_rgb_depth_skew_s').value)
        with self._lock:
            for name, buf in self._buffers.items():
                if any(value is None for value in (buf.depth, buf.rgb, buf.k,
                                                    buf.depth_stamp, buf.rgb_stamp)):
                    continue
                if abs(buf.depth_stamp - buf.rgb_stamp) > allowed:
                    continue
                self._sync.update(name, CameraSample(buf.depth.copy(), buf.rgb.copy(),
                                                     buf.k.copy(), buf.depth_stamp))
        return self._sync.take()

    def _mount(self, name):
        translation = self.get_parameter(f'mounts.{name}.translation').value
        yaw = float(self.get_parameter(f'mounts.{name}.yaw').value)
        return translation, yaw

    def _process(self):
        samples = self._collect()
        stale_limit = float(self.get_parameter('stale_camera_s').value)
        stale = [name for name, wall in self._last_input_wall.items()
                 if not wall or time.monotonic() - wall > stale_limit]
        if samples is None:
            reason = 'stale=' + ','.join(stale) if stale else 'waiting_for_synchronized_set'
            self._status_pub.publish(String(data='NOT_READY:' + reason))
            return
        clouds = []
        for name, sample in samples.items():
            translation, yaw = self._mount(name)
            clouds.append(depth_to_base_points(
                sample.depth, sample.intrinsic, translation, yaw,
                stride=int(self.get_parameter('depth_stride').value),
                min_depth=float(self.get_parameter('min_depth_m').value),
                max_depth=float(self.get_parameter('max_depth_m').value)))
        points = np.concatenate(clouds, axis=0) if clouds else np.empty((0, 3))
        stamp_s = min(sample.stamp_s for sample in samples.values())
        layers = derive_layers(
            self._spec, points, samples['front'].bgr,
            [samples[name].bgr for name in ('front', 'left', 'right')],
            self._separator, self._tracker, self._voxel_grid, stamp_s,
            float(self.get_parameter('inflation_radius_m').value))
        for name, grid in layers.items():
            self._publishers[name].publish(self._grid_message(grid, stamp_s))
        skew_ms = (max(s.stamp_s for s in samples.values()) - stamp_s) * 1000
        self._status_pub.publish(String(data=f'ACTIVE:points={len(points)}:skew_ms={skew_ms:.1f}'))

    def _grid_message(self, grid, stamp_s):
        msg = OccupancyGrid()
        seconds = int(stamp_s)
        msg.header.stamp.sec = seconds
        msg.header.stamp.nanosec = int((stamp_s - seconds) * 1e9)
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.info.resolution = self._spec.resolution
        msg.info.width, msg.info.height = self._spec.shape[1], self._spec.shape[0]
        msg.info.origin.position.x = -self._spec.width_m / 2
        msg.info.origin.position.y = -self._spec.height_m / 2
        msg.info.origin.orientation.w = 1.0
        msg.data = np.asarray(grid, dtype=np.int8).ravel().tolist()
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = ThreeZedPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
