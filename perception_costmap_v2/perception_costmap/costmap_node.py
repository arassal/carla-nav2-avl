"""
costmap_node.py -- the ONLY module in this package that imports rclpy.

Wires N camera sources (+ one optional lidar) through the ROS-free modules
(segmentation, obstacles, bev, temporal, occupancy) into a published
nav_msgs/OccupancyGrid + sensor_msgs/PointCloud2, parameterized entirely
from config/perception_costmap.yaml.

This file is deliberately "thin": every piece of logic that could plausibly
be unit-tested without ROS lives in the sibling modules. What's here is
subscription plumbing, staleness bookkeeping, per-tick fusion order, and
publishing -- exactly the part that only makes sense with a running graph.

Sim-to-real: only the sensor *source* changes between CARLA and the Jetson
(same topics, same message types, same QoS). This node's code does not
change at all.
"""

import numpy as np

from .occupancy import (GridSpec, build_cost_array, to_occupancy_grid_msg,
                        UNKNOWN, DEFAULT_OBSTACLE_CLASSES)
from .bev import warp_to_bev, bev_known_mask
from .segmentation import create_segmenter
from .obstacles import (points_to_grid_mask, remove_ground_plane,
                        YoloObstacleDetector, detect_obstacles_classical,
                        camera_obstacle_mask_to_grid)
from .temporal import TemporalObstacleFilter
from .util import stamp_to_sec, is_fresh, image_msg_to_bgr, pointcloud2_to_xyz


class CameraSource:
    """Per-camera state: last image/known-mask + the homography that warps
    this camera's masks into the shared grid. One instance per entry in the
    `cameras: [...]` config list; costmap_node loops these each tick."""

    def __init__(self, name: str, H: np.ndarray, grid: GridSpec,
                image_shape, max_age: float = 0.5, image_topic: str = None):
        self.name = name
        self.H = H
        self.grid = grid
        self.image_shape = image_shape
        self.known_mask = bev_known_mask(H, image_shape, grid)
        self.max_age = max_age
        self.image_topic = image_topic or ("/camera/%s/image" % name)
        self.last_stamp_sec = None
        self.last_image = None

    def on_image(self, img_bgr, stamp_sec: float):
        self.last_image = img_bgr
        self.last_stamp_sec = stamp_sec

    def is_fresh(self, now_sec: float) -> bool:
        if self.last_stamp_sec is None:
            return False
        return is_fresh(self.last_stamp_sec, now_sec, self.max_age)


class CostmapNode:
    """
    ROS-graph-shaped but constructible/testable without rclpy: __init__
    takes plain config, and `_tick(now_sec)` is a pure function of the
    accumulated CameraSource/lidar state, returning a cost array. The rclpy
    subscription callbacks (added in `attach_ros`) just call `on_image` /
    `on_lidar` and are the only rclpy-touching code in this class.
    """

    def __init__(self, grid: GridSpec, cameras: dict,
                segmentation_method="hsv", segmentation_kw=None,
                obstacle_method="classical", yolo_kw=None,
                lidar_z_min=0.15, lidar_z_max=3.0,
                temporal_enabled=True, temporal_kw=None,
                offroad_cost=None, road_edge_radius=1.5,
                unknown_infill=True, lidar_topic="/lidar/points",
                publish_rate_hz=10.0):
        self.grid = grid
        self.cameras = cameras  # name -> CameraSource
        self.segmenter = create_segmenter(segmentation_method, **(segmentation_kw or {}))
        self.obstacle_method = obstacle_method
        self.yolo = YoloObstacleDetector(**(yolo_kw or {})) if obstacle_method in ("yolo", "both") else None
        self.lidar_z_min = lidar_z_min
        self.lidar_z_max = lidar_z_max
        self.temporal = TemporalObstacleFilter((grid.height, grid.width), **(temporal_kw or {})) if temporal_enabled else None
        self.offroad_cost = offroad_cost
        self.road_edge_radius = road_edge_radius
        self.unknown_infill = unknown_infill
        self.lidar_topic = lidar_topic
        self.publish_rate_hz = publish_rate_hz
        self._last_lidar_points = np.zeros((0, 3))
        self._last_lidar_stamp_sec = None
        self._lidar_max_age = 0.5
        self._node = None
        self._subs = []

    def on_lidar(self, points_xyz: np.ndarray, stamp_sec: float):
        self._last_lidar_points = remove_ground_plane(points_xyz, self.lidar_z_min, self.lidar_z_max)
        self._last_lidar_stamp_sec = stamp_sec

    def _camera_obstacle_mask(self, cam: CameraSource) -> np.ndarray:
        img = cam.last_image
        if self.obstacle_method == "classical":
            mask_img = detect_obstacles_classical(img)
        elif self.obstacle_method == "yolo":
            mask_img = self.yolo.detect(img)
        elif self.obstacle_method == "both":
            mask_img = detect_obstacles_classical(img) | self.yolo.detect(img)
        else:
            mask_img = np.zeros(img.shape[:2], dtype=bool)
        # Clip to this camera's own known footprint -- see
        # obstacles.camera_obstacle_mask_to_grid's docstring: unclipped, a
        # perspective-warp "mirror cell" behind the camera can plant a
        # spurious obstacle far from anything the camera actually saw.
        return camera_obstacle_mask_to_grid(mask_img, cam.H, self.grid, known_mask=cam.known_mask)

    def _tick(self, now_sec: float) -> np.ndarray:
        """Fuse every fresh camera + lidar into one cost array. Returns the
        int8 cost array (height, width); costmap_node.publish wraps it."""
        shape = (self.grid.height, self.grid.width)
        road = np.zeros(shape, dtype=bool)
        cam_obstacle = np.zeros(shape, dtype=bool)
        known = np.zeros(shape, dtype=bool)
        any_camera_fresh = False

        for cam in self.cameras.values():
            if not cam.is_fresh(now_sec):
                continue
            any_camera_fresh = True
            road_img_mask = self.segmenter(cam.last_image)
            road_grid_mask = camera_obstacle_mask_to_grid(road_img_mask, cam.H, self.grid,
                                                           known_mask=cam.known_mask)
            road |= road_grid_mask
            cam_obstacle |= self._camera_obstacle_mask(cam)
            known |= cam.known_mask

        lidar_fresh = (self._last_lidar_stamp_sec is not None and
                      is_fresh(self._last_lidar_stamp_sec, now_sec, self._lidar_max_age))

        if not any_camera_fresh and not lidar_fresh:
            # Blind tick: publish "I know nothing" rather than the last thing
            # we believed. Without this the TemporalObstacleFilter latches --
            # it only decays confidence on cells it OBSERVED this tick, and a
            # blind tick observes none, so the final obstacle set is republished
            # unchanged forever (measured: 6.69% of cells still LETHAL, bit
            # identical, 40s after the sensors stopped). Nav2 handles an
            # unknown costmap; it cannot detect a confidently wrong one.
            #
            # Temporal confidence is deliberately NOT reset. Recovery from a
            # brief dropout then keeps its history, and obstacles that really
            # did disappear decay normally once cells are observed again.
            return np.full(shape, UNKNOWN, dtype=np.int8)

        lidar_obstacle = np.zeros(shape, dtype=bool)
        if lidar_fresh:
            lidar_obstacle = points_to_grid_mask(self._last_lidar_points, self.grid)
            # Lidar alone still lets a tick publish: if every camera is
            # stale, everything stays UNKNOWN except lidar obstacles, rather
            # than skipping the tick outright.
            if not any_camera_fresh:
                known = known | lidar_obstacle  # lidar-seen cells are "known"

        fused_obstacle_raw = cam_obstacle | lidar_obstacle

        if self.temporal is not None:
            observed = known if any_camera_fresh else lidar_obstacle
            fused_obstacle = self.temporal.update(fused_obstacle_raw, observed)
        else:
            fused_obstacle = fused_obstacle_raw

        kw = dict(road_edge_radius=self.road_edge_radius,
                  unknown_infill=self.unknown_infill)
        if self.offroad_cost is not None:
            kw["offroad_cost"] = self.offroad_cost

        return build_cost_array(self.grid, road, fused_obstacle, known_mask=known, **kw)

    # ---- rclpy wiring (only touched when actually running as a ROS node) ----

    def attach_ros(self, node):
        """Create subscriptions/publishers on an existing rclpy Node and a
        wall-timer that calls _tick and publishes. Kept out of __init__ so
        this class stays constructible (and _tick callable) with no rclpy
        installed at all -- that's what makes the fusion logic testable."""
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image, PointCloud2
        from nav_msgs.msg import OccupancyGrid

        # BEST_EFFORT/volatile: CARLA's ROS2 output and real camera/lidar
        # drivers publish sensor-data QoS. Subscribing RELIABLE (the rclpy
        # default) silently receives zero messages on many DDS pairings --
        # documented as the #1 sim-to-real bug in this package's history.
        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        sensor_qos.durability = DurabilityPolicy.VOLATILE

        self._costmap_pub = node.create_publisher(OccupancyGrid, "/perception/costmap", 1)
        self._node = node

        for cam in self.cameras.values():
            self._subs.append(node.create_subscription(
                Image, cam.image_topic, self._make_image_cb(cam), sensor_qos))
            node.get_logger().info("camera %r <- %s" % (cam.name, cam.image_topic))

        self._subs.append(node.create_subscription(
            PointCloud2, self.lidar_topic, self._on_lidar_msg, sensor_qos))
        node.get_logger().info("lidar <- %s" % self.lidar_topic)

        period = 1.0 / float(self.publish_rate_hz)
        self._timer = node.create_timer(period, self._on_timer)
        return self._costmap_pub

    def _now_sec(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _make_image_cb(self, cam: "CameraSource"):
        def cb(msg):
            # A decode failure must not kill the executor: one malformed
            # frame (or a camera that changed encoding mid-run) would
            # otherwise take down the whole costmap. Drop it, say so once
            # per occurrence, and let staleness handle the consequences.
            try:
                img = image_msg_to_bgr(msg)
            except ValueError as e:
                self._node.get_logger().warn("camera %s: %s" % (cam.name, e))
                return
            if img.shape[:2] != tuple(cam.image_shape[:2]):
                # known_mask was precomputed for the configured resolution;
                # silently accepting another one would misplace every cell.
                self._node.get_logger().warn(
                    "camera %s: got %dx%d but configured for %dx%d -- ignoring "
                    "frame (fix the config or the driver)"
                    % (cam.name, img.shape[1], img.shape[0],
                       cam.image_shape[1], cam.image_shape[0]))
                return
            cam.on_image(img, stamp_to_sec(msg.header.stamp))
        return cb

    def _on_lidar_msg(self, msg):
        try:
            pts = pointcloud2_to_xyz(msg)
        except ValueError as e:
            self._node.get_logger().warn("lidar: %s" % e)
            return
        self.on_lidar(pts, stamp_to_sec(msg.header.stamp))

    def _on_timer(self):
        self.publish(self._now_sec())

    def publish(self, now_sec: float):
        cost = self._tick(now_sec)
        # Stamp it. Consumers (Nav2's StaticLayer, rviz, any TF lookup into
        # base_link) treat a default 0/0 header.stamp as epoch-old -- the same
        # failure the camera feed had on the input side, just one hop later.
        stamp = self._node.get_clock().now().to_msg() if self._node else None
        msg = to_occupancy_grid_msg(cost, self.grid, stamp=stamp)
        self._costmap_pub.publish(msg)
        return cost


def build_from_params(node):
    """Construct a CostmapNode from an rclpy Node's declared parameters
    (config/perception_costmap.yaml). Split out from main() so a launch file
    or a test harness can build the node without owning the spin loop."""
    from .bev import homography_from_extrinsics, homography_from_points

    def p(name, default):
        # main() creates the Node with
        # automatically_declare_parameters_from_overrides=True so that the
        # nested `front.*` blocks in the YAML arrive at all; anything the
        # user actually set is therefore already declared, and re-declaring
        # it raises. Declare only what's missing, so unset keys still get
        # the defaults below.
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
        value = node.get_parameter(name).value
        return default if value is None else value

    grid = GridSpec(
        x_min=p("grid_x_min", -4.0), x_max=p("grid_x_max", 16.0),
        y_min=p("grid_y_min", -10.0), y_max=p("grid_y_max", 10.0),
        resolution=p("grid_resolution", 0.1),
    )

    image_w = p("image_width", 640)
    image_h = p("image_height", 480)
    image_shape = (image_h, image_w)

    cameras = {}
    for name in p("cameras", ["front"]):
        def cp(key, default):
            return p("%s.%s" % (name, key), default)
        fx = cp("fx", 320.0)
        fy = cp("fy", fx)
        K = np.array([[fx, 0.0, cp("cx", image_w / 2.0)],
                      [0.0, fy, cp("cy", image_h / 2.0)],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        if cp("ipm_mode", "camera") == "points":
            image_pts = np.array(cp("ipm_image_pts", [0.0] * 8), dtype=np.float64).reshape(4, 2)
            world_pts = np.array(cp("ipm_world_pts", [0.0] * 8), dtype=np.float64).reshape(4, 2)
            H = homography_from_points(image_pts, world_pts, grid)
        else:
            H = homography_from_extrinsics(
                K, (cp("cam_x", 1.5), cp("cam_y", 0.0), cp("cam_height", 1.6)),
                cp("cam_pitch_deg", 0.0), cp("cam_yaw_deg", 0.0), grid)
        cameras[name] = CameraSource(
            name, H, grid, image_shape,
            max_age=cp("max_age_sec", 0.5),
            image_topic=cp("image_topic", "/camera/%s/image" % name),
        )

    seg_kw = dict(
        lower_hsv=tuple(p("segmentation_lower_hsv", [0, 0, 60])),
        upper_hsv=tuple(p("segmentation_upper_hsv", [180, 60, 200])),
        min_blob_area=p("segmentation_min_blob_area", 500),
    )

    return CostmapNode(
        grid=grid, cameras=cameras,
        segmentation_method=p("segmentation_method", "hsv"),
        segmentation_kw=seg_kw,
        obstacle_method=p("obstacle_method", "classical"),
        lidar_z_min=p("lidar_z_min", 0.15),
        lidar_z_max=p("lidar_z_max", 3.0),
        temporal_enabled=p("temporal_enabled", True),
        temporal_kw=dict(hit=p("temporal_hit", 0.4),
                         miss=p("temporal_miss", 0.2),
                         threshold=p("temporal_threshold", 0.5)),
        offroad_cost=p("offroad_cost", 65),
        road_edge_radius=p("road_edge_radius", 1.5),
        unknown_infill=p("unknown_infill", True),
        lidar_topic=p("lidar_topic", "/lidar/points"),
        publish_rate_hz=p("publish_rate_hz", 10.0),
    )


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("perception_costmap_node",
                automatically_declare_parameters_from_overrides=True)
    costmap = build_from_params(node)
    costmap.attach_ros(node)
    node.get_logger().info("publishing /perception/costmap at %.1f Hz"
                           % costmap.publish_rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        # SIGINT delivered by launch/systemd rather than a bare Ctrl-C:
        # rclpy has already torn the context down. Without this the process
        # exits with a traceback, which reads as a crash in journalctl.
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
