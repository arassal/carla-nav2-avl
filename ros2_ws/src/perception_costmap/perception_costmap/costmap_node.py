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


class CameraSource:
    """One camera: its subscriptions, latest frame, homography and FOV mask.
    All parameters live under '<name>.' so a 3-camera car is 3 YAML blocks."""

    def __init__(self, node, name, grid):
        self.name, self.grid = name, grid
        d = lambda key, val: node.declare_parameter("%s.%s" % (name, key), val).value
        self.ipm_mode = d("ipm_mode", "points")
        self.image_pts = np.array(d("ipm_image_pts",
            [0.0, 160.0, 640.0, 160.0, 640.0, 320.0, 0.0, 320.0]), float).reshape(4, 2)
        self.world_pts = np.array(d("ipm_world_pts",
            [18.0, 8.0, 18.0, -8.0, 3.0, -4.0, 3.0, 4.0]), float).reshape(4, 2)
        self.cam_xyz = (d("cam_x", 0.0), d("cam_y", 0.0), d("cam_height", 1.6))
        self.pitch = d("cam_pitch_deg", 10.0)
        self.yaw = d("cam_yaw_deg", 0.0)
        self.img, self.stamp, self.K = None, 0.0, None
        self.H, self.known = None, None
        self._node = node
        node.create_subscription(Image, d("image_topic", "/camera/%s/image" % name),
                                 self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo,
                                 d("camera_info_topic", "/camera/%s/camera_info" % name),
                                 self._on_info, qos_profile_sensor_data)

    def _on_image(self, msg):
        if self._node._bridge is None:
            from cv_bridge import CvBridge
            self._node._bridge = CvBridge()
        self.img = self._node._bridge.imgmsg_to_cv2(msg, "bgr8")
        self.stamp = stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def ensure_homography(self):
        if self.H is not None:
            return True
        if self.ipm_mode == "camera":
            if self.K is None:
                return False
            self.H = bev.homography_from_extrinsics(
                self.K, self.cam_xyz, self.pitch, self.yaw, self.grid)
        else:
            self.H = bev.homography_from_points(
                self.image_pts, self.world_pts, self.grid)
        self.known = bev.bev_known_mask(self.H, self.img.shape, self.grid)
        return True


class CostmapNode(Node):
    def __init__(self):
        super().__init__("perception_costmap")

        # ---- parameters ----
        p = self.declare_parameters("", [
            ("cameras", ["front"]),
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
            # obstacle_method: classical = no deps, yolo = accurate classes +
            # Jetson .engine path, both = union
            ("obstacle_method", "classical"),
            ("yolo_weights", "yolov8n.pt"),
            ("yolo_conf", 0.35),
            # course classes (IGVC): dedicated cone detector + painted
            # white-line boundary. See FINALIZE.md Phase 1.
            ("use_cones", False),
            # cameras the COCO detector runs on ([] = all). IGVC: front only
            # (side coverage = cones + 360 lidar) buys ~25 ms/tick
            ("yolo_cameras", [""]),
            ("cone_weights", "cone_det.pt"),
            ("cone_conf", 0.35),
            ("use_white_lines", False),
            ("yolo_footprint_frac", 0.25),
            ("twinlite_repo_path", ""),
            ("twinlite_weights", ""),
            ("twinlite_config", "nano"),
            ("lidar_z_min", 0.2), ("lidar_z_max", 2.5),
            # freshness budgets (sec); with use_sim_time set, node clock and
            # CARLA stamps share the same timeline
            ("image_stale_sec", 0.5), ("lidar_stale_sec", 0.5),
            # temporal obstacle confidence filter
            ("temporal_hit", 0.4), ("temporal_miss", 0.2),
            ("temporal_threshold", 0.5), ("temporal_enabled", True),
            # cost shape: off-road (observed, no obstacle) vs. true obstacle,
            # plus a distance-decay halo around obstacles so the costmap
            # isn't just a flat lethal/free binary. See occupancy.py.
            ("offroad_cost", 65),
            ("inflation_radius", 0.8),
            ("cost_scaling_factor", 4.0),
            # -1 = ROS unknown; small positive = blind spots traversable
            # with a mild penalty (see occupancy.build_cost_array)
            ("unknown_cost", -1),
        ])
        g = {k.name: k.value for k in p}

        self.grid = GridSpec(
            x_min=g["x_min"], x_max=g["x_max"],
            y_min=g["y_min"], y_max=g["y_max"],
            resolution=g["resolution"], frame_id=g["frame_id"],
        )
        self.use_cam_obs = g["use_camera_obstacles"]
        self.use_lidar = g["use_lidar"]
        self.obstacle_method = g["obstacle_method"]
        self.z_min, self.z_max = g["lidar_z_min"], g["lidar_z_max"]
        self.img_stale, self.lidar_stale = g["image_stale_sec"], g["lidar_stale_sec"]
        self.temporal_enabled = g["temporal_enabled"]
        self.offroad_cost = g["offroad_cost"]
        self.inflation_radius = g["inflation_radius"]
        self.cost_scaling_factor = g["cost_scaling_factor"]
        self.unknown_cost = g["unknown_cost"]
        self.obs_filter = TemporalObstacleFilter(
            (self.grid.height, self.grid.width),
            hit=g["temporal_hit"], miss=g["temporal_miss"],
            threshold=g["temporal_threshold"],
        )

        # models must warm-load at startup, never mid-drive
        self.yolo = None
        if g["obstacle_method"] in ("yolo", "both"):
            try:
                self.yolo = obstacles.YoloObstacleDetector(
                    weights=g["yolo_weights"], conf=g["yolo_conf"],
                    footprint_frac=g["yolo_footprint_frac"])
                self.get_logger().info("YOLO obstacle detector loaded: %s" % g["yolo_weights"])
            except Exception as e:
                self.get_logger().warn(
                    "YOLO unavailable (%s); falling back to classical" % e)

        self.yolo_cams = set(c for c in g["yolo_cameras"] if c)
        self.cones = None
        if g["use_cones"]:
            try:
                self.cones = obstacles.ConeDetector(
                    weights=g["cone_weights"], conf=g["cone_conf"])
                self.get_logger().info(
                    "cone detector loaded: %s" % g["cone_weights"])
            except Exception as e:
                self.get_logger().warn(
                    "cones unavailable (%s); disabled" % e)
        self.use_white_lines = bool(g["use_white_lines"])

        try:
            if g["segmentation_method"] == "twinlitenet":
                self.segmenter = segmentation.create_segmenter(
                    "twinlitenet", repo_path=g["twinlite_repo_path"],
                    weights=g["twinlite_weights"], config=g["twinlite_config"])
            else:
                self.segmenter = segmentation.create_segmenter("hsv")
        except Exception as e:                     # missing torch/weights/paths
            self.get_logger().warn("twinlitenet unavailable (%s); using hsv" % e)
            self.segmenter = segmentation.create_segmenter("hsv")

        self._bridge = None          # cv_bridge, created lazily
        self._latest_points = None   # (N,3) ndarray
        self._pts_stamp = None       # seconds, header stamp of latest lidar scan

        self.cameras = [CameraSource(self, n, self.grid) for n in g["cameras"]]

        # ---- pub/sub ----
        self.costmap_pub = self.create_publisher(OccupancyGrid, g["costmap_topic"], 1)
        self.obs_pub = self.create_publisher(PointCloud2, g["obstacle_points_topic"], 1)
        if self.use_lidar:
            self.create_subscription(
                PointCloud2, g["lidar_topic"], self._on_lidar, qos_profile_sensor_data)

        self.create_timer(1.0 / float(g["publish_rate"]), self._tick)
        self.get_logger().info(
            f"perception_costmap up: {self.grid.width}x{self.grid.height} "
            f"@ {self.grid.resolution} m/cell, cameras={g['cameras']}")

    # ---- callbacks ----
    def _on_lidar(self, msg):
        from sensor_msgs_py import point_cloud2
        # read_points (structured), not read_points_numpy: the latter
        # asserts all cloud fields share one dtype and dies on real velodyne
        # clouds (float32 xyz/intensity + uint16 ring) -- found 2026-07-07,
        # first time live velodyne data reached this callback
        arr = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)
        self._latest_points = np.stack(
            [np.asarray(arr[k], float) for k in ("x", "y", "z")], axis=-1)
        self._pts_stamp = stamp_to_sec(msg.header.stamp)

    # ---- main loop ----
    def _tick(self):
        now = stamp_to_sec(self.get_clock().now().to_msg())
        empty = np.zeros((self.grid.height, self.grid.width), bool)
        road_bev = empty.copy()
        obst_grid = empty.copy()
        known = np.zeros((self.grid.height, self.grid.width), bool)
        saw_camera = False

        for cam in self.cameras:
            if cam.img is None or not is_fresh(cam.stamp, now, self.img_stale):
                continue
            if not cam.ensure_homography():
                continue
            saw_camera = True
            road = self.segmenter(cam.img)
            if self.use_white_lines:
                # painted course lines are boundaries, not drivable
                road = road & ~segmentation.white_line_mask(cam.img)
            # clip to the camera's footprint: warpPerspective also fills
            # mirror cells behind the camera plane (negative projective depth)
            road_bev |= (bev.warp_to_bev(
                road.astype(np.uint8) * 255, cam.H, self.grid) > 127) & cam.known
            known |= cam.known
            if self.use_cam_obs:
                obs_img = np.zeros(cam.img.shape[:2], bool)
                if self.obstacle_method in ("classical", "both") or self.yolo is None:
                    obs_img |= obstacles.detect_obstacles_camera(cam.img, road)
                if self.yolo is not None and (
                        not self.yolo_cams or cam.name in self.yolo_cams):
                    obs_img |= self.yolo.detect(cam.img)
                if self.cones is not None:
                    obs_img |= self.cones.detect(cam.img)
                obst_grid |= (bev.warp_to_bev(
                    obs_img.astype(np.uint8) * 255, cam.H, self.grid) > 127) & cam.known

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
        # tick means "observed" is everything; otherwise it's the union of
        # camera FOVs
        lidar_active = (self.use_lidar and self._latest_points is not None
                        and is_fresh(self._pts_stamp, now, self.lidar_stale))
        observed = known
        if lidar_active:
            observed = np.ones_like(observed)
        if self.temporal_enabled:
            obst_grid = self.obs_filter.update(obst_grid, observed)

        if not saw_camera and not lidar_active:  # nothing seen yet
            return

        cost = build_cost_array(
            self.grid, road_bev, obst_grid, known_mask=known,
            offroad_cost=self.offroad_cost,
            inflation_radius=self.inflation_radius,
            cost_scaling_factor=self.cost_scaling_factor,
            unknown_cost=self.unknown_cost)
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
