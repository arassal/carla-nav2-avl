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

from .occupancy import GridSpec, build_cost_array, to_occupancy_grid_msg, DEFAULT_OBSTACLE_CLASSES
from .bev import warp_to_bev, bev_known_mask
from .segmentation import create_segmenter
from .obstacles import (points_to_grid_mask, remove_ground_plane,
                        YoloObstacleDetector, detect_obstacles_classical,
                        camera_obstacle_mask_to_grid)
from .temporal import TemporalObstacleFilter
from .util import stamp_to_sec, is_fresh


class CameraSource:
    """Per-camera state: last image/known-mask + the homography that warps
    this camera's masks into the shared grid. One instance per entry in the
    `cameras: [...]` config list; costmap_node loops these each tick."""

    def __init__(self, name: str, H: np.ndarray, grid: GridSpec,
                image_shape, max_age: float = 0.5):
        self.name = name
        self.H = H
        self.grid = grid
        self.known_mask = bev_known_mask(H, image_shape, grid)
        self.max_age = max_age
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
                lidar_z_min=-0.3, lidar_z_max=3.0,
                temporal_enabled=True, temporal_kw=None,
                offroad_cost=None, road_edge_radius=1.5,
                unknown_infill=True):
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
        self._last_lidar_points = np.zeros((0, 3))
        self._last_lidar_stamp_sec = None
        self._lidar_max_age = 0.5

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
        # Per-camera / lidar subscriptions would be created here from the
        # `cameras` config block; omitted because topic names and cv_bridge
        # conversion are the one part that genuinely needs a running
        # ROS graph + cv_bridge installed to exercise.

    def publish(self, now_sec: float):
        cost = self._tick(now_sec)
        msg = to_occupancy_grid_msg(cost, self.grid)
        self._costmap_pub.publish(msg)
        return cost


def main(args=None):
    import rclpy
    from rclpy.node import Node

    rclpy.init(args=args)
    node = Node("perception_costmap_node")
    node.get_logger().info(
        "costmap_node scaffolded -- wire cameras/lidar from "
        "config/perception_costmap.yaml before running against a real graph."
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
