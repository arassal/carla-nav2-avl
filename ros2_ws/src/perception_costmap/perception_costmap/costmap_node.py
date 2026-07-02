"""
costmap_node.py
---------------
The ROS2 perception node. Subscribes to a forward camera (and optionally a
lidar), runs road segmentation + obstacle detection, projects them into a
top-down metric grid, and publishes:

  /perception/costmap          nav_msgs/OccupancyGrid    (road + obstacles)
  /perception/obstacle_points  sensor_msgs/PointCloud2   (lidar obstacles)

Everything is parameterised (see config/perception_costmap.yaml). The heavy
lifting lives in the ROS-free modules (segmentation, obstacles, bev,
occupancy); this file is just the ROS plumbing.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import OccupancyGrid

from .occupancy import GridSpec, build_cost_array, to_occupancy_grid_msg
from . import segmentation, obstacles, bev
from .util import stamp_to_sec, is_fresh
from .temporal import TemporalObstacleFilter


class CostmapNode(Node):
    def __init__(self):
        super().__init__("perception_costmap")

        # ---- parameters ----
        p = self.declare_parameters("", [
            ("image_topic", "/camera/image"),
            ("camera_info_topic", "/camera/camera_info"),
            ("lidar_topic", "/lidar/points"),
            ("costmap_topic", "/perception/costmap"),
            ("obstacle_points_topic", "/perception/obstacle_points"),
            ("publish_rate", 10.0),
            ("frame_id", "base_link"),
            ("x_min", -4.0), ("x_max", 16.0),
            ("y_min", -10.0), ("y_max", 10.0),
            ("resolution", 0.1),
            ("segmentation_method", "hsv"),
            ("use_camera_obstacles", True),
            ("use_lidar", True),
            ("lidar_z_min", 0.2), ("lidar_z_max", 2.5),
            # IPM: "points" uses image_pts/world_pts; "camera" uses K+mounting
            ("ipm_mode", "points"),
            ("ipm_image_pts", [0.0, 160.0, 640.0, 160.0, 640.0, 320.0, 0.0, 320.0]),
            ("ipm_world_pts", [18.0, -8.0, 18.0, 8.0, 3.0, 4.0, 3.0, -4.0]),
            ("cam_height", 1.6), ("cam_pitch_deg", 10.0),
            # freshness budgets (sec); with use_sim_time set, node clock and
            # CARLA stamps share the same timeline
            ("image_stale_sec", 0.5), ("lidar_stale_sec", 0.5),
            # temporal obstacle confidence filter
            ("temporal_hit", 0.4), ("temporal_miss", 0.2),
            ("temporal_threshold", 0.5), ("temporal_enabled", True),
        ])
        g = {k.name: k.value for k in p}

        self.grid = GridSpec(
            x_min=g["x_min"], x_max=g["x_max"],
            y_min=g["y_min"], y_max=g["y_max"],
            resolution=g["resolution"], frame_id=g["frame_id"],
        )
        self.seg_method = g["segmentation_method"]
        self.use_cam_obs = g["use_camera_obstacles"]
        self.use_lidar = g["use_lidar"]
        self.z_min, self.z_max = g["lidar_z_min"], g["lidar_z_max"]
        self.ipm_mode = g["ipm_mode"]
        self.ipm_image_pts = np.array(g["ipm_image_pts"], float).reshape(4, 2)
        self.ipm_world_pts = np.array(g["ipm_world_pts"], float).reshape(4, 2)
        self.cam_height, self.cam_pitch = g["cam_height"], g["cam_pitch_deg"]
        self.img_stale, self.lidar_stale = g["image_stale_sec"], g["lidar_stale_sec"]
        self.temporal_enabled = g["temporal_enabled"]
        self.obs_filter = TemporalObstacleFilter(
            (self.grid.height, self.grid.width),
            hit=g["temporal_hit"], miss=g["temporal_miss"],
            threshold=g["temporal_threshold"],
        )

        self._bridge = None          # cv_bridge, created lazily
        self._latest_img = None      # bgr ndarray
        self._latest_points = None   # (N,3) ndarray
        self._img_stamp = None       # seconds, header stamp of latest image
        self._pts_stamp = None       # seconds, header stamp of latest lidar scan
        self._K = None               # camera intrinsics (camera mode)
        self._H = None               # cached image->grid homography
        self._known = None           # cached FOV footprint

        # ---- pub/sub ----
        self.costmap_pub = self.create_publisher(OccupancyGrid, g["costmap_topic"], 1)
        self.obs_pub = self.create_publisher(PointCloud2, g["obstacle_points_topic"], 1)
        self.create_subscription(
            Image, g["image_topic"], self._on_image, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, g["camera_info_topic"], self._on_info, qos_profile_sensor_data)
        if self.use_lidar:
            self.create_subscription(
                PointCloud2, g["lidar_topic"], self._on_lidar, qos_profile_sensor_data)

        self.create_timer(1.0 / float(g["publish_rate"]), self._tick)
        self.get_logger().info(
            f"perception_costmap up: {self.grid.width}x{self.grid.height} "
            f"@ {self.grid.resolution} m/cell, ipm={self.ipm_mode}")

    # ---- callbacks ----
    def _on_image(self, msg):
        if self._bridge is None:
            from cv_bridge import CvBridge
            self._bridge = CvBridge()
        self._latest_img = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        self._img_stamp = stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg):
        self._K = np.array(msg.k, float).reshape(3, 3)

    def _on_lidar(self, msg):
        from sensor_msgs_py import point_cloud2
        pts = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True)
        self._latest_points = np.asarray(pts, float).reshape(-1, 3)
        self._pts_stamp = stamp_to_sec(msg.header.stamp)

    # ---- homography (built once we have what we need) ----
    def _ensure_homography(self, image_shape):
        if self._H is not None:
            return True
        if self.ipm_mode == "camera":
            if self._K is None:
                return False
            self._H = bev.homography_from_camera(
                self._K, self.cam_height, self.cam_pitch, self.grid)
        else:
            self._H = bev.homography_from_points(
                self.ipm_image_pts, self.ipm_world_pts, self.grid)
        self._known = bev.bev_known_mask(self._H, image_shape, self.grid)
        return True

    # ---- main loop ----
    def _tick(self):
        now = stamp_to_sec(self.get_clock().now().to_msg())
        empty = np.zeros((self.grid.height, self.grid.width), bool)
        road_bev = empty
        obst_grid = empty.copy()
        known = None

        img_fresh = (
            self._latest_img is not None
            and is_fresh(self._img_stamp, now, self.img_stale)
        )
        if img_fresh and self._ensure_homography(self._latest_img.shape):
            road = segmentation.segment_road(self._latest_img, method=self.seg_method)
            road_bev = bev.warp_to_bev(road.astype(np.uint8) * 255, self._H, self.grid) > 127
            known = self._known
            if self.use_cam_obs:
                obs_img = obstacles.detect_obstacles_camera(self._latest_img, road)
                obst_grid |= bev.warp_to_bev(
                    obs_img.astype(np.uint8) * 255, self._H, self.grid) > 127

        lidar_fresh = (
            self._latest_points is not None
            and is_fresh(self._pts_stamp, now, self.lidar_stale)
        )
        if self.use_lidar and lidar_fresh:
            pts = obstacles.filter_obstacle_points(
                self._latest_points, self.z_min, self.z_max)
            obst_grid |= obstacles.points_to_grid_mask(pts, self.grid)
            self._publish_obstacle_points(pts)

        # a 360° lidar observes the whole grid, so a fresh lidar frame this
        # tick means "observed" is everything; otherwise it's the camera FOV
        lidar_active = (self.use_lidar and self._latest_points is not None
                        and is_fresh(self._pts_stamp, now, self.lidar_stale))
        observed = (known if known is not None
                    else np.zeros((self.grid.height, self.grid.width), bool))
        if lidar_active:
            observed = np.ones_like(observed)
        if self.temporal_enabled:
            obst_grid = self.obs_filter.update(obst_grid, observed)

        if known is None:                       # nothing seen yet
            return

        cost = build_cost_array(self.grid, road_bev, obst_grid, known_mask=known)
        msg = to_occupancy_grid_msg(cost, self.grid, stamp=self.get_clock().now().to_msg())
        self.costmap_pub.publish(msg)

    def _publish_obstacle_points(self, pts):
        from sensor_msgs_py import point_cloud2
        from std_msgs.msg import Header
        hdr = Header()
        hdr.stamp = self.get_clock().now().to_msg()
        hdr.frame_id = self.grid.frame_id
        self.obs_pub.publish(point_cloud2.create_cloud_xyz32(hdr, pts.tolist()))


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
