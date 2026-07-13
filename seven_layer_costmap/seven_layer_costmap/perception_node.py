"""Three-ZED perception node deriving six costmap layers from RGB/depth only."""

import math
import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from .core import GridSpec
from .perception import (CameraSample, CentroidTracker, Pose2D,
                         SkewMonitor, ThreeCameraSynchronizer, depth_to_base_points,
                         derive_layers, inflation_radius_for_speed, stamp_seconds,
                         WorldOccupancyModel)


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
        self.declare_parameter('processing_frequency', 5.0)
        self.declare_parameter('depth_stride', 4)
        self.declare_parameter('min_depth_m', 0.5)
        self.declare_parameter('max_depth_m', 25.0)
        self.declare_parameter('width_m', 60.0)
        self.declare_parameter('height_m', 60.0)
        self.declare_parameter('resolution', 0.20)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('inflation_radius_m', 2.5)
        self.declare_parameter('inflation_reaction_time_s', 0.35)
        self.declare_parameter('inflation_max_speed_extra_m', 3.0)
        self.declare_parameter('voxel_persistence_s', 2.0)
        self.declare_parameter('max_clear_rays_per_cycle', 75)
        self.declare_parameter('visibility_max_rays_per_cycle', 1200)
        self.declare_parameter('visibility_dilation_cells', 1)
        self.declare_parameter('blind_spot_centers_deg', [-45.0, 45.0])
        self.declare_parameter('blind_spot_half_width_deg', 18.0)
        self.declare_parameter('blind_spot_min_range_m', 1.5)
        self.declare_parameter('blind_spot_max_range_m', 12.0)
        self.declare_parameter('blind_spot_unknown_cost', 25)
        self.declare_parameter('blind_spot_clear_cost', 0)
        self.declare_parameter('require_odometry', True)
        self.declare_parameter('ego_min_x', -2.5)
        self.declare_parameter('ego_max_x', 0.8)
        self.declare_parameter('ego_half_width', 1.2)
        self.declare_parameter('timestamp_offsets_s.front', 0.0)
        self.declare_parameter('timestamp_offsets_s.left', 0.0)
        self.declare_parameter('timestamp_offsets_s.right', 0.0)
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
        self._skew_monitor = SkewMonitor(float(self.get_parameter('max_camera_skew_s').value))
        self._spec = GridSpec(float(self.get_parameter('width_m').value),
                              float(self.get_parameter('height_m').value),
                              float(self.get_parameter('resolution').value))
        self._tracker = CentroidTracker()
        self._occupancy = WorldOccupancyModel(
            self._spec, persistence_s=float(self.get_parameter('voxel_persistence_s').value),
            max_clear_rays=int(self.get_parameter('max_clear_rays_per_cycle').value))
        self._pose = None
        self._pose_stamp = None
        self._speed_mps = 0.0
        self._accepted_sets = 0
        self._rejected_sets = 0
        self._last_input_wall = {name: 0.0 for name in self._logical}
        self._layer_publishers = {name: self.create_publisher(
            OccupancyGrid, f'/seven_layer_costmap/layers/{name}', 1)
            for name in ('lanelet', 'static_obstacle', 'spatio_temporal_voxel',
                         'prediction', 'inflation', 'traffic_regulation')}
        self._status_pub = self.create_publisher(String, '/seven_layer_costmap/perception_status', 10)
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, '/seven_layer_costmap/diagnostics', 10)
        for logical, camera in self._logical.items():
            prefix = f'/{camera}/{camera}_node'
            self.create_subscription(Image, prefix + '/depth/depth_registered',
                                     lambda m, n=logical: self._depth(n, m), qos_profile_sensor_data)
            self.create_subscription(Image, prefix + '/left/color/rect/image',
                                     lambda m, n=logical: self._rgb(n, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, prefix + '/left/camera_info',
                                     lambda m, n=logical: self._info(n, m), qos_profile_sensor_data)
        front_camera = self._logical['front']
        self.create_subscription(Odometry, f'/{front_camera}/{front_camera}_node/odom',
                                 self._odom, qos_profile_sensor_data)
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

    def _odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        p = msg.pose.pose.position
        with self._lock:
            self._pose = Pose2D(p.x, p.y, yaw)
            self._pose_stamp = (stamp_seconds(msg.header.stamp) + float(
                self.get_parameter('timestamp_offsets_s.front').value))
            velocity = msg.twist.twist.linear
            self._speed_mps = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def _collect(self):
        allowed = float(self.get_parameter('max_rgb_depth_skew_s').value)
        with self._lock:
            for name, buf in self._buffers.items():
                if any(value is None for value in (buf.depth, buf.rgb, buf.k,
                                                    buf.depth_stamp, buf.rgb_stamp)):
                    continue
                if abs(buf.depth_stamp - buf.rgb_stamp) > allowed:
                    continue
                corrected = buf.depth_stamp + float(
                    self.get_parameter(f'timestamp_offsets_s.{name}').value)
                self._sync.update(name, CameraSample(buf.depth.copy(), buf.rgb.copy(),
                                                     buf.k.copy(), corrected))
            if all(name in self._sync.samples for name in self._logical):
                self._skew_monitor.observe(
                    self._sync.samples[name].stamp_s for name in self._logical)
        return self._sync.take()

    def _mount(self, name):
        translation = self.get_parameter(f'mounts.{name}.translation').value
        yaw = float(self.get_parameter(f'mounts.{name}.yaw').value)
        return translation, yaw

    def _process(self):
        try:
            self._process_once()
        except Exception as error:  # Keep the timer alive and fail closed on malformed input.
            self._rejected_sets += 1
            self.get_logger().error(
                f'Perception processing failed: {type(error).__name__}: {error}',
                throttle_duration_sec=2.0)
            self._publish_health(DiagnosticStatus.ERROR,
                                 f'PROCESSING_ERROR:{type(error).__name__}',
                                 0, time.monotonic())

    def _process_once(self):
        started = time.monotonic()
        samples = self._collect()
        stale_limit = float(self.get_parameter('stale_camera_s').value)
        stale = [name for name, wall in self._last_input_wall.items()
                 if not wall or time.monotonic() - wall > stale_limit]
        if samples is None:
            self._rejected_sets += 1
            reason = 'stale=' + ','.join(stale) if stale else 'waiting_for_synchronized_set'
            self._publish_health(DiagnosticStatus.WARN, 'NOT_READY:' + reason, 0, started)
            return
        with self._lock:
            pose, pose_stamp = self._pose, self._pose_stamp
        if bool(self.get_parameter('require_odometry').value):
            if pose is None or pose_stamp is None or abs(min(
                    sample.stamp_s for sample in samples.values()) - pose_stamp) > 0.25:
                self._rejected_sets += 1
                self._publish_health(DiagnosticStatus.ERROR,
                                     'NOT_READY:missing_or_stale_odometry', 0, started)
                return
        pose = pose or Pose2D()
        clouds = []
        origins = []
        for name, sample in samples.items():
            translation, yaw = self._mount(name)
            cloud = depth_to_base_points(
                sample.depth, sample.intrinsic, translation, yaw,
                stride=int(self.get_parameter('depth_stride').value),
                min_depth=float(self.get_parameter('min_depth_m').value),
                max_depth=float(self.get_parameter('max_depth_m').value))
            # Reject points on the ego vehicle before occupancy or ray processing.
            inside_ego = ((cloud[:, 0] >= float(self.get_parameter('ego_min_x').value)) &
                          (cloud[:, 0] <= float(self.get_parameter('ego_max_x').value)) &
                          (np.abs(cloud[:, 1]) <= float(self.get_parameter('ego_half_width').value)))
            cloud = cloud[~inside_ego]
            clouds.append(cloud)
            origins.append(np.repeat(np.asarray(translation, dtype=np.float32)[None, :],
                                     len(cloud), axis=0))
        points = np.concatenate(clouds, axis=0) if clouds else np.empty((0, 3))
        sensor_origins = np.concatenate(origins, axis=0) if origins else np.empty((0, 3))
        stamp_s = min(sample.stamp_s for sample in samples.values())
        inflation_radius = inflation_radius_for_speed(
            float(self.get_parameter('inflation_radius_m').value), self._speed_mps,
            float(self.get_parameter('inflation_reaction_time_s').value),
            float(self.get_parameter('inflation_max_speed_extra_m').value))
        layers = derive_layers(
            self._spec, points, samples['front'].bgr,
            [samples[name].bgr for name in ('front', 'left', 'right')],
            self._occupancy, self._tracker, pose, stamp_s, sensor_origins,
            inflation_radius,
            visibility_max_rays=int(
                self.get_parameter('visibility_max_rays_per_cycle').value),
            visibility_dilation_cells=int(
                self.get_parameter('visibility_dilation_cells').value),
            blind_centers_deg=list(
                self.get_parameter('blind_spot_centers_deg').value),
            blind_half_width_deg=float(
                self.get_parameter('blind_spot_half_width_deg').value),
            blind_min_range_m=float(
                self.get_parameter('blind_spot_min_range_m').value),
            blind_max_range_m=float(
                self.get_parameter('blind_spot_max_range_m').value),
            blind_unknown_cost=int(
                self.get_parameter('blind_spot_unknown_cost').value),
            blind_clear_cost=int(
                self.get_parameter('blind_spot_clear_cost').value))
        for name, grid in layers.items():
            self._layer_publishers[name].publish(self._grid_message(grid, stamp_s))
        self._accepted_sets += 1
        self._publish_health(DiagnosticStatus.OK, 'ACTIVE', len(points), started,
                             inflation_radius)

    def _publish_health(self, level, message, points, started, inflation_radius=0.0):
        elapsed_ms = (time.monotonic() - started) * 1000
        skew = self._skew_monitor.summary()
        text = (f'{message}:points={points}:skew_ms={skew["current_s"] * 1000:.1f}:'
                f'latency_ms={elapsed_ms:.1f}')
        self._status_pub.publish(String(data=text))
        status = DiagnosticStatus()
        status.level = level
        status.name = 'seven_layer_costmap/three_zed_perception'
        status.hardware_id = 'three_zed_svo'
        status.message = message
        values = {
            'points': points, 'current_skew_ms': skew['current_s'] * 1000,
            'mean_skew_ms': skew['mean_s'] * 1000,
            'max_skew_ms': skew['max_s'] * 1000,
            'skew_violations': skew['violations'], 'processing_latency_ms': elapsed_ms,
            'accepted_sets': self._accepted_sets, 'rejected_sets': self._rejected_sets,
            'speed_mps': self._speed_mps, 'inflation_radius_m': inflation_radius,
            'active_voxels': len(self._occupancy.last_seen),
        }
        status.values = [KeyValue(key=str(key), value=str(value))
                         for key, value in values.items()]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self._diagnostic_pub.publish(array)

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
