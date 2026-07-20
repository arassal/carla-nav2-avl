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
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String

from .core import GridSpec
from .perception import (CameraSample, CentroidTracker, Pose2D,
                         SkewMonitor, ThreeCameraSynchronizer, colorize_bev,
                         depth_to_base_points, derive_layers,
                         inflation_radius_for_speed, stamp_seconds,
                         vision_bev_grid, WorldOccupancyModel)


class CameraBuffer:
    def __init__(self):
        self.depth = self.rgb = self.k = None
        self.depth_stamp = self.rgb_stamp = None
        self.last_enqueued_stamp = None


class ThreeZedPerceptionNode(Node):
    def __init__(self):
        super().__init__('three_zed_perception')
        self.declare_parameter('camera_names', ['zed_front', 'zed_left', 'zed_right'])
        self.declare_parameter('max_camera_skew_s', 0.050)
        self.declare_parameter('max_rgb_depth_skew_s', 0.035)
        self.declare_parameter('sync_queue_size', 90)
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
        self.declare_parameter('use_motion_compensation', False)
        self.declare_parameter('enable_temporal_memory', False)
        self.declare_parameter('enable_prediction', False)
        self.declare_parameter('bev_ground_band_m', 0.18)
        self.declare_parameter('bev_obstacle_z_min', -1.5)
        self.declare_parameter('bev_obstacle_z_max', 2.8)
        self.declare_parameter('bev_min_points_per_cell', 2)
        self.declare_parameter('ego_min_x', -2.5)
        self.declare_parameter('ego_max_x', 0.8)
        self.declare_parameter('ego_half_width', 1.2)
        self.declare_parameter('timestamp_offsets_s.front', 0.0)
        self.declare_parameter('timestamp_offsets_s.left', 0.0)
        self.declare_parameter('timestamp_offsets_s.right', 0.0)
        # Provisional sketch-derived mount values. Override after calibration.
        self.declare_parameter('mounts.front.translation', [0.67945, 0.0, -0.10795])
        self.declare_parameter('mounts.front.rpy', [0.0, 0.0, 0.0])
        self.declare_parameter('mounts.left.translation', [0.098425, -0.28575, 0.05715])
        self.declare_parameter('mounts.left.rpy', [0.0, 0.0, math.pi / 2])
        self.declare_parameter('mounts.right.translation', [0.098425, 0.28575, 0.05715])
        self.declare_parameter('mounts.right.rpy', [0.0, 0.0, -math.pi / 2])

        camera_names = list(self.get_parameter('camera_names').value)
        if len(camera_names) != 3:
            raise ValueError('exactly three camera_names are required')
        self._logical = dict(zip(('front', 'left', 'right'), camera_names))
        self._bridge = CvBridge()
        self._buffers = {logical: CameraBuffer() for logical in self._logical}
        self._lock = threading.Lock()
        self._sync = ThreeCameraSynchronizer(max_skew_s=float(
            self.get_parameter('max_camera_skew_s').value), queue_size=int(
                self.get_parameter('sync_queue_size').value))
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
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._bev_pub = self.create_publisher(
            OccupancyGrid, '/seven_layer_costmap/bev/occupancy', map_qos)
        self._bev_image_pub = self.create_publisher(
            Image, '/seven_layer_costmap/bev/image', qos_profile_sensor_data)
        self._cloud_pub = self.create_publisher(
            PointCloud2, '/seven_layer_costmap/points/fused', qos_profile_sensor_data)
        self._status_pub = self.create_publisher(String, '/seven_layer_costmap/perception_status', 10)
        self._diagnostic_pub = self.create_publisher(
            DiagnosticArray, '/seven_layer_costmap/diagnostics', 10)
        for logical, camera in self._logical.items():
            prefix = f'/{camera}/{camera}_node'
            self.create_subscription(Image, prefix + '/depth/depth_registered',
                                     lambda m, n=logical: self._depth(n, m), qos_profile_sensor_data)
            self.create_subscription(Image, prefix + '/rgb/color/rect/image',
                                     lambda m, n=logical: self._rgb(n, m), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, prefix + '/rgb/color/rect/camera_info',
                                     lambda m, n=logical: self._info(n, m), qos_profile_sensor_data)
        if bool(self.get_parameter('use_motion_compensation').value):
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
                self._enqueue_camera_locked(name)
        except Exception as error:
            self.get_logger().warn(f'{name} depth decode: {error}', throttle_duration_sec=5.0)

    def _rgb(self, name, msg):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self._lock:
                self._buffers[name].rgb = rgb
                self._buffers[name].rgb_stamp = stamp_seconds(msg.header.stamp)
                self._enqueue_camera_locked(name)
        except Exception as error:
            self.get_logger().warn(f'{name} RGB decode: {error}', throttle_duration_sec=5.0)

    def _info(self, name, msg):
        with self._lock:
            self._buffers[name].k = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
            self._enqueue_camera_locked(name)

    def _enqueue_camera_locked(self, name):
        """Pair one camera's RGB/depth and queue it for three-way timestamp sync."""
        buf = self._buffers[name]
        if any(value is None for value in (buf.depth, buf.rgb, buf.k,
                                            buf.depth_stamp, buf.rgb_stamp)):
            return
        if abs(buf.depth_stamp - buf.rgb_stamp) > float(
                self.get_parameter('max_rgb_depth_skew_s').value):
            return
        corrected = buf.depth_stamp + float(
            self.get_parameter(f'timestamp_offsets_s.{name}').value)
        if buf.last_enqueued_stamp is not None and corrected <= buf.last_enqueued_stamp:
            return
        self._sync.update(name, CameraSample(
            buf.depth.copy(), buf.rgb.copy(), buf.k.copy(), corrected))
        buf.last_enqueued_stamp = corrected

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
        with self._lock:
            samples = self._sync.take()
            if samples is not None:
                self._skew_monitor.observe(
                    samples[name].stamp_s for name in self._logical)
            return samples

    def _mount(self, name):
        translation = self.get_parameter(f'mounts.{name}.translation').value
        rpy = self.get_parameter(f'mounts.{name}.rpy').value
        return translation, rpy

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
        motion_compensation = bool(self.get_parameter('use_motion_compensation').value)
        if motion_compensation:
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
            translation, rpy = self._mount(name)
            cloud = depth_to_base_points(
                sample.depth, sample.intrinsic, translation, rpy=rpy,
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
        speed = self._speed_mps if motion_compensation else 0.0
        inflation_radius = inflation_radius_for_speed(
            float(self.get_parameter('inflation_radius_m').value), speed,
            float(self.get_parameter('inflation_reaction_time_s').value),
            float(self.get_parameter('inflation_max_speed_extra_m').value))
        bev, bev_obstacles = vision_bev_grid(
            self._spec, points, sensor_origins,
            visibility_max_rays=int(
                self.get_parameter('visibility_max_rays_per_cycle').value),
            visibility_dilation_cells=int(
                self.get_parameter('visibility_dilation_cells').value),
            ground_band_m=float(self.get_parameter('bev_ground_band_m').value),
            obstacle_z_min=float(self.get_parameter('bev_obstacle_z_min').value),
            obstacle_z_max=float(self.get_parameter('bev_obstacle_z_max').value),
            min_points_per_cell=int(
                self.get_parameter('bev_min_points_per_cell').value))
        self._bev_pub.publish(self._grid_message(bev, stamp_s))
        bev_image = self._bridge.cv2_to_imgmsg(colorize_bev(bev), 'bgr8')
        bev_image.header.stamp = self._stamp_message(stamp_s)
        bev_image.header.frame_id = self.get_parameter('frame_id').value
        self._bev_image_pub.publish(bev_image)
        self._cloud_pub.publish(self._cloud_message(points, stamp_s))
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
                self.get_parameter('blind_spot_clear_cost').value),
            temporal_memory=bool(
                self.get_parameter('enable_temporal_memory').value),
            enable_prediction=bool(
                self.get_parameter('enable_prediction').value))
        for name, grid in layers.items():
            self._layer_publishers[name].publish(self._grid_message(grid, stamp_s))
        self._accepted_sets += 1
        self._publish_health(DiagnosticStatus.OK, 'ACTIVE', len(points), started,
                             inflation_radius, len(bev_obstacles))

    def _publish_health(self, level, message, points, started, inflation_radius=0.0,
                        bev_obstacles=0):
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
            'sync_queue_drops': self._sync.dropped,
            'speed_mps': self._speed_mps, 'inflation_radius_m': inflation_radius,
            'active_voxels': len(self._occupancy.last_seen),
            'bev_obstacle_points': bev_obstacles,
            'motion_compensation': bool(
                self.get_parameter('use_motion_compensation').value),
            'temporal_memory': bool(
                self.get_parameter('enable_temporal_memory').value),
        }
        status.values = [KeyValue(key=str(key), value=str(value))
                         for key, value in values.items()]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status = [status]
        self._diagnostic_pub.publish(array)

    def _grid_message(self, grid, stamp_s):
        msg = OccupancyGrid()
        msg.header.stamp = self._stamp_message(stamp_s)
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.info.resolution = self._spec.resolution
        msg.info.width, msg.info.height = self._spec.shape[1], self._spec.shape[0]
        msg.info.origin.position.x = -self._spec.width_m / 2
        msg.info.origin.position.y = -self._spec.height_m / 2
        msg.info.origin.orientation.w = 1.0
        msg.data = np.asarray(grid, dtype=np.int8).ravel().tolist()
        return msg

    @staticmethod
    def _stamp_message(stamp_s):
        from builtin_interfaces.msg import Time
        nanoseconds = int(round(float(stamp_s) * 1e9))
        seconds, remainder = divmod(nanoseconds, 1_000_000_000)
        return Time(sec=seconds, nanosec=remainder)

    def _cloud_message(self, points, stamp_s):
        xyz = np.ascontiguousarray(np.asarray(points, dtype='<f4').reshape(-1, 3))
        msg = PointCloud2()
        msg.header.stamp = self._stamp_message(stamp_s)
        msg.header.frame_id = self.get_parameter('frame_id').value
        msg.height = 1
        msg.width = len(xyz)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.data = xyz.tobytes()
        msg.is_dense = bool(np.isfinite(xyz).all())
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = ThreeZedPerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
