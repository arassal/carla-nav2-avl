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

import cv2
import numpy as np

import os
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import OccupancyGrid, Odometry
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from std_srvs.srv import Trigger

from .occupancy import GridSpec, build_cost_array, to_occupancy_grid_msg
from . import segmentation, obstacles, bev
from .util import stamp_to_sec, is_fresh, clear_sample_buffer
from .temporal import TemporalObstacleFilter, remap_with, reproject_maps
from .detection_schedule import DetectionScheduler
from .inference_worker import LatestTaskWorker
from .sample_buffer import PoseBuffer, TimestampedBuffer
from .health import camera_health


class CameraSource:
    """One camera: its subscriptions, latest frame, homography and FOV mask.
    All parameters live under '<name>.' so a 3-camera car is 3 YAML blocks."""

    def __init__(self, node, name, grid, fov_edge_trim_m=0.4):
        self.name, self.grid = name, grid
        self.fov_edge_trim_m = fov_edge_trim_m
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
        self.depth_buffer = TimestampedBuffer(maxlen=4)
        self.confidence_buffer = TimestampedBuffer(maxlen=4)
        self.last_yolo_stamp = None
        self.last_cone_stamp = None
        self.H, self.known = None, None
        self._node = node
        node.create_subscription(Image, d("image_topic", "/camera/%s/image" % name),
                                 self._on_image, qos_profile_sensor_data)
        node.create_subscription(CameraInfo,
                                 d("camera_info_topic", "/camera/%s/camera_info" % name),
                                 self._on_info, qos_profile_sensor_data)
        depth_topic = d("depth_topic", "")
        if depth_topic:
            node.create_subscription(
                Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        confidence_topic = d("confidence_topic", "")
        self.confidence_expected = bool(confidence_topic)
        if confidence_topic:
            node.create_subscription(
                Image, confidence_topic, self._on_confidence,
                qos_profile_sensor_data)

    def _on_image(self, msg):
        if self._node._bridge is None:
            from cv_bridge import CvBridge
            self._node._bridge = CvBridge()
        self.img = self._node._bridge.imgmsg_to_cv2(msg, "bgr8")
        self.stamp = stamp_to_sec(msg.header.stamp)

    def _on_info(self, msg):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def _on_depth(self, msg):
        if self._node._bridge is None:
            from cv_bridge import CvBridge
            self._node._bridge = CvBridge()
        depth = self._node._bridge.imgmsg_to_cv2(
            msg, desired_encoding="32FC1").astype(np.float32, copy=False)
        self.depth_buffer.add(stamp_to_sec(msg.header.stamp), depth)

    def _on_confidence(self, msg):
        if self._node._bridge is None:
            from cv_bridge import CvBridge
            self._node._bridge = CvBridge()
        confidence = self._node._bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough").astype(np.float32, copy=False)
        self.confidence_buffer.add(stamp_to_sec(msg.header.stamp), confidence)

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
        known = bev.bev_known_mask(self.H, self.img.shape, self.grid)
        # Trim the FOV border: segmentation is unreliable in the outermost
        # image pixels, so the strip of BEV ground right at a camera's
        # coverage edge kept classifying as "observed, not road" -> off-road
        # 97 -> phantom red bands hugging every blind seam. Eroding the mask
        # makes that strip UNKNOWN instead, and the blind-spot infill guesses
        # it from its (usually drivable) neighbourhood.
        trim_m = float(self.fov_edge_trim_m)
        if trim_m > 0:
            r = max(1, int(round(trim_m / self.grid.resolution)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1, 2*r+1))
            known = cv2.erode(known.astype(np.uint8), kernel).astype(bool)
        self.known = known
        return True


class CostmapNode(Node):
    def __init__(self):
        super().__init__("perception_costmap")

        # ---- parameters ----
        p = self.declare_parameters("", [
            ("cameras", ["front"]),
            ("required_cameras", ["front"]),
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
            ("require_yolo", False),
            # course classes (IGVC): dedicated cone detector + painted
            # white-line boundary. See FINALIZE.md Phase 1.
            ("use_cones", False),
            # Cameras the detectors run on ([] = all). Scheduling keeps the
            # primary camera current while bounding side-camera GPU load.
            ("yolo_cameras", [""]),
            ("cone_cameras", [""]),
            ("primary_detection_camera", "front"),
            ("secondary_detection_stride", 1),
            ("secondary_cone_stride", 4),
            ("cone_weights", "cone_det.pt"),
            ("cone_conf", 0.35),
            ("require_cones", False),
            ("use_white_lines", False),
            ("yolo_footprint_frac", 0.25),
            ("twinlite_repo_path", ""),
            ("twinlite_weights", ""),
            ("twinlite_config", "nano"),
            ("lidar_z_min", 0.2), ("lidar_z_max", 2.5),
            # freshness budgets (sec); with use_sim_time set, node clock and
            # CARLA stamps share the same timeline
            ("image_stale_sec", 0.5), ("lidar_stale_sec", 0.5),
            ("depth_sync_tolerance_sec", 0.15),
            ("depth_wait_sec", 0.08),
            ("depth_min_m", 0.3), ("depth_max_m", 20.0),
            ("depth_obstacle_dilation_m", 0.15),
            ("depth_confidence_max", 70.0),
            ("depth_edge_erode_px", 1),
            ("depth_mad_scale", 3.5),
            ("depth_relative_tolerance", 0.12),
            # temporal obstacle confidence filter
            ("temporal_hit", 0.4), ("temporal_miss", 0.2),
            ("temporal_threshold", 0.5), ("temporal_enabled", True),
            ("motion_compensation", True),
            ("odometry_topic", "/wheel_odom"),
            ("odometry_stale_sec", 0.5),
            ("odometry_interpolation_tolerance_sec", 0.1),
            ("detector_stale_sec", 0.5),
            # cost shape: off-road (observed, no obstacle) vs. true obstacle,
            # plus a distance-decay halo around obstacles so the costmap
            # isn't just a flat lethal/free binary. See occupancy.py.
            ("offroad_cost", 65),
            ("inflation_radius", 0.8),
            ("cost_scaling_factor", 4.0),
            # inward ramp from the road edge: cost rises the closer you get to
            # leaving the road, instead of a flat step at the boundary
            ("road_edge_radius", 1.5),
            ("road_edge_scaling", 2.0),
            ("min_offroad_width_m", 0.0),
            # blind-spot infill: guess unobserved cells from surrounding
            # observed ground, degrading to unknown_cost with distance
            ("unknown_infill", True),
            ("infill_falloff", 2.0),
            # trim each camera's BEV coverage inward: segmentation is junk at
            # the image border, which painted phantom off-road bands along
            # every blind-seam edge
            ("fov_edge_trim_m", 0.4),
            # per-class danger zones (radius m, exponential decay rate)
            ("person_radius", 2.5),
            ("person_scaling", 1.5),
            ("person_exclusion_radius", 1.2),
            ("vehicle_radius", 1.5),
            ("vehicle_scaling", 2.5),
            ("vehicle_exclusion_radius", 0.6),
            ("cone_radius", 0.6),
            ("cone_scaling", 5.0),
            ("cone_exclusion_radius", 0.2),
            ("generic_radius", 1.0),
            ("generic_scaling", 3.0),
            ("generic_exclusion_radius", 0.5),
            # Vulnerable road users become lethal on one strong observation
            # and clear more cautiously. Generic blobs retain two-hit
            # confirmation to suppress segmentation noise.
            ("person_temporal_hit", 0.6),
            ("person_temporal_miss", 0.2),
            ("person_temporal_threshold", 0.5),
            ("vehicle_temporal_hit", 0.5),
            ("vehicle_temporal_miss", 0.25),
            ("vehicle_temporal_threshold", 0.5),
            # -1 = ROS unknown; small positive = blind spots traversable
            # with a mild penalty (see occupancy.build_cost_array)
            ("unknown_cost", -1),
        ])
        g = {k.name: k.value for k in p}
        # Portable model paths: expand ${AVL_MODELS_DIR} and ~ so the
        # committed config works on any machine (see models/README.md).
        # Bare filenames / empty values pass through unchanged.
        for _k in ("yolo_weights", "cone_weights",
                   "twinlite_weights", "twinlite_repo_path"):
            if g.get(_k):
                g[_k] = os.path.expanduser(os.path.expandvars(g[_k]))

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
        self.required_cameras = set(g["required_cameras"])
        self.depth_sync = float(g["depth_sync_tolerance_sec"])
        self.depth_wait = float(g["depth_wait_sec"])
        self.depth_min = float(g["depth_min_m"])
        self.depth_max = float(g["depth_max_m"])
        self.depth_dilation = float(g["depth_obstacle_dilation_m"])
        self.depth_confidence_max = float(g["depth_confidence_max"])
        self.depth_edge_erode_px = int(g["depth_edge_erode_px"])
        self.depth_mad_scale = float(g["depth_mad_scale"])
        self.depth_relative_tolerance = float(g["depth_relative_tolerance"])
        self.temporal_enabled = g["temporal_enabled"]
        self.motion_compensation = bool(g["motion_compensation"])
        self.odom_stale = float(g["odometry_stale_sec"])
        self.odom_interpolation_tolerance = float(
            g["odometry_interpolation_tolerance_sec"])
        self.detector_stale = float(g["detector_stale_sec"])
        self.offroad_cost = g["offroad_cost"]
        self.inflation_radius = g["inflation_radius"]
        self.cost_scaling_factor = g["cost_scaling_factor"]
        self.unknown_cost = g["unknown_cost"]
        self.road_edge_radius = g["road_edge_radius"]
        self.road_edge_scaling = g["road_edge_scaling"]
        self.min_offroad_width_m = g["min_offroad_width_m"]
        self.unknown_infill = g["unknown_infill"]
        self.infill_falloff = g["infill_falloff"]
        self.obstacle_classes = {
            "person": dict(
                radius=g["person_radius"], scaling=g["person_scaling"],
                exclusion_radius=g["person_exclusion_radius"]),
            "vehicle": dict(
                radius=g["vehicle_radius"], scaling=g["vehicle_scaling"],
                exclusion_radius=g["vehicle_exclusion_radius"]),
            "cone": dict(
                radius=g["cone_radius"], scaling=g["cone_scaling"],
                exclusion_radius=g["cone_exclusion_radius"]),
            "generic": dict(
                radius=g["generic_radius"], scaling=g["generic_scaling"],
                exclusion_radius=g["generic_exclusion_radius"]),
        }
        temporal_specs = {
            "person": (
                g["person_temporal_hit"], g["person_temporal_miss"],
                g["person_temporal_threshold"]),
            "vehicle": (
                g["vehicle_temporal_hit"], g["vehicle_temporal_miss"],
                g["vehicle_temporal_threshold"]),
            "cone": (
                g["temporal_hit"], g["temporal_miss"],
                g["temporal_threshold"]),
            "generic": (
                g["temporal_hit"], g["temporal_miss"],
                g["temporal_threshold"]),
        }
        self.obs_filters = {
            group: TemporalObstacleFilter(
                (self.grid.height, self.grid.width),
                hit=temporal_specs[group][0], miss=temporal_specs[group][1],
                threshold=temporal_specs[group][2])
            for group in temporal_specs
        }

        # models must warm-load at startup, never mid-drive
        self.yolo = None
        if g["obstacle_method"] in ("yolo", "both"):
            try:
                self.yolo = obstacles.YoloObstacleDetector(
                    weights=g["yolo_weights"], conf=g["yolo_conf"],
                    footprint_frac=g["yolo_footprint_frac"])
                self.get_logger().info("YOLO obstacle detector loaded: %s" % g["yolo_weights"])
            except Exception as e:
                if g["require_yolo"]:
                    raise RuntimeError(
                        "required YOLO detector failed to load: %s" % e) from e
                self.get_logger().warn(
                    "YOLO unavailable (%s); falling back to classical" % e)

        self.yolo_cams = set(c for c in g["yolo_cameras"] if c)
        self.cone_cams = set(c for c in g["cone_cameras"] if c)
        self.detection_scheduler = DetectionScheduler(
            primary=g["primary_detection_camera"],
            secondary_stride=g["secondary_detection_stride"])
        self.cone_scheduler = DetectionScheduler(
            primary=g["primary_detection_camera"],
            secondary_stride=g["secondary_cone_stride"])
        self.cones = None
        if g["use_cones"]:
            try:
                self.cones = obstacles.ConeDetector(
                    weights=g["cone_weights"], conf=g["cone_conf"])
                self.get_logger().info(
                    "cone detector loaded: %s" % g["cone_weights"])
            except Exception as e:
                if g["require_cones"]:
                    raise RuntimeError(
                        "required cone detector failed to load: %s" % e) from e
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
        self._odom_pose = None
        self._odom_stamp = None
        self._odom_buffer = PoseBuffer(maxlen=100)
        self._filter_pose = None
        self._filter_odom_stamp = None
        self._depth_projections = 0
        self._ipm_fallbacks = 0
        self._confidence_projections = 0
        self._confidence_missing = 0
        self._confidence_rejected = 0
        self._depth_outlier_rejected = 0
        self._depth_matches = 0
        self._depth_unmatched = 0
        self._depth_waits = 0
        self._ticks = 0
        self._have_detection_result = False
        self._last_detection_result_time = None
        self._last_publish_time = None
        self._detector_errors = 0
        self._last_detector_error_time = None
        self._last_inference_cameras = {"yolo": [], "cones": []}
        self._require_detection_result = bool(
            (g["require_yolo"] and self.yolo is not None)
            or (g["require_cones"] and self.cones is not None))

        self.cameras = [CameraSource(self, n, self.grid,
                                     fov_edge_trim_m=g["fov_edge_trim_m"])
                        for n in g["cameras"]]
        self.detector_worker = LatestTaskWorker(self._process_detection_task)

        # ---- pub/sub ----
        self.costmap_pub = self.create_publisher(OccupancyGrid, g["costmap_topic"], 1)
        # Publish the observed-ground mask alongside the costmap. Consumers
        # (rviz colouring, analysis) must NOT recompute coverage themselves:
        # with unknown_cost>0 the grid value cannot distinguish "unknown" from
        # a real cost, and any second implementation of the coverage test
        # drifts from this one (it did -- blind cells got painted as low-cost
        # "go" ground). 100 = observed, 0 = never seen.
        self.known_pub = self.create_publisher(OccupancyGrid, "/perception/known", 1)
        self.obs_pub = self.create_publisher(PointCloud2, g["obstacle_points_topic"], 1)
        self.health_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10)
        # Between-runs reset. The temporal filters accumulate evidence across
        # ticks by design and nothing else clears them, so a second run in the
        # same process starts with the first run's obstacles still confirmed.
        # IGVC requires each run carry nothing over; restarting the whole stack
        # did this by accident. See deploy/fresh_run.sh.
        self.create_service(Trigger, "/perception/reset", self._on_reset)
        if self.use_lidar:
            self.create_subscription(
                PointCloud2, g["lidar_topic"], self._on_lidar, qos_profile_sensor_data)
        if self.motion_compensation:
            self.create_subscription(
                Odometry, g["odometry_topic"], self._on_odom,
                qos_profile_sensor_data)

        self.create_timer(1.0 / float(g["publish_rate"]), self._tick)
        self.create_timer(1.0, self._publish_diagnostics)
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

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._odom_pose = (p.x, p.y, yaw)
        self._odom_stamp = stamp_to_sec(msg.header.stamp)
        self._odom_buffer.add(self._odom_stamp, self._odom_pose)

    def _project_obstacle_snapshot(self, camera, image_mask):
        depth_ready = (
            camera["depth"] is not None and camera["K"] is not None
            and abs(camera["depth_stamp"] - camera["stamp"]) <= self.depth_sync)
        if depth_ready:
            projected, stats = obstacles.depth_mask_to_grid(
                image_mask, camera["depth"], camera["K"], camera["cam_xyz"],
                camera["pitch"], camera["yaw"], self.grid,
                min_depth=self.depth_min, max_depth=self.depth_max,
                dilation_m=self.depth_dilation,
                confidence=camera["confidence"],
                max_confidence=self.depth_confidence_max,
                edge_erode_px=self.depth_edge_erode_px,
                depth_mad_scale=self.depth_mad_scale,
                depth_relative_tolerance=self.depth_relative_tolerance,
                return_stats=True)
            if projected is not None:
                return projected, True, stats
        return (((bev.warp_to_bev(
            image_mask.astype(np.uint8) * 255, camera["H"], self.grid) > 127)
                 & camera["known"]), False, stats if depth_ready else None)

    def _process_detection_task(self, task):
        """Run all GPU detectors serially outside the ROS timer thread."""
        empty = np.zeros((self.grid.height, self.grid.width), bool)
        observations = []
        depth_count = 0
        ipm_count = 0
        confidence_count = 0
        confidence_missing = 0
        confidence_rejected = 0
        outlier_rejected = 0
        ran_yolo = []
        ran_cones = []

        for camera in task["cameras"]:
            camera_grids = {
                group: empty.copy()
                for group in ("person", "vehicle", "cone", "generic")
            }
            class_images = {}
            if camera["run_classical"]:
                class_images["generic"] = obstacles.detect_obstacles_camera(
                    camera["image"], camera["road"])
            if camera["run_yolo"]:
                for group, mask in self.yolo.detect_grouped(
                        camera["image"]).items():
                    class_images[group] = (
                        class_images.get(group, np.zeros_like(mask)) | mask)
                ran_yolo.append(camera["name"])
            if camera["run_cones"]:
                class_images["cone"] = self.cones.detect(camera["image"])
                ran_cones.append(camera["name"])

            for group, image_mask in class_images.items():
                if group not in camera_grids:
                    continue
                projected, used_depth, stats = self._project_obstacle_snapshot(
                    camera, image_mask)
                camera_grids[group] |= projected
                depth_count += int(used_depth)
                ipm_count += int(not used_depth)
                if used_depth and stats is not None:
                    confidence_count += int(stats["confidence_available"])
                    confidence_missing += int(not stats["confidence_available"])
                    confidence_rejected += stats["confidence_rejected"]
                    outlier_rejected += stats["outlier_rejected"]

            observations.append({
                "class_grids": camera_grids,
                "observed": camera["known"],
                "pose": camera["pose"],
                "stamp": camera["stamp"],
            })

        return {
            "observations": observations,
            "depth_count": depth_count,
            "ipm_count": ipm_count,
            "confidence_count": confidence_count,
            "confidence_missing": confidence_missing,
            "confidence_rejected": confidence_rejected,
            "outlier_rejected": outlier_rejected,
            "ran_yolo": ran_yolo,
            "ran_cones": ran_cones,
        }

    def _consume_detection_result(self, now):
        error = self.detector_worker.take_error()
        if error is not None:
            self._detector_errors += 1
            self._last_detector_error_time = now
            self.get_logger().error("detector inference failed: %s" % error)

        result = self.detector_worker.take_latest()
        if result is None:
            return None

        empty = np.zeros((self.grid.height, self.grid.width), bool)
        class_grids = {
            group: empty.copy()
            for group in ("person", "vehicle", "cone", "generic")
        }
        observed = empty.copy()
        accepted = 0
        for observation in result["observations"]:
            if not is_fresh(observation["stamp"], now, self.img_stale):
                continue
            masks = observation["class_grids"]
            seen = observation["observed"]
            if (observation["pose"] is not None and self._odom_pose is not None
                    and is_fresh(self._odom_stamp, now, self.odom_stale)):
                # One pose change, so one set of sampling maps -- build them
                # once and reuse for `seen` plus every class mask. Calling
                # reproject_grid per array rebuilt these identical maps 5x per
                # observation, which profiled as ~10% of main-thread time.
                maps = reproject_maps(
                    seen.shape, observation["pose"], self._odom_pose,
                    self.grid)
                seen = remap_with(seen.astype(np.uint8), maps).astype(bool)
                masks = {
                    group: remap_with(mask.astype(np.uint8), maps).astype(bool)
                    for group, mask in masks.items()
                }
            observed |= seen
            for group, mask in masks.items():
                class_grids[group] |= mask
            accepted += 1
        if accepted == 0:
            return None

        self._have_detection_result = True
        self._last_detection_result_time = now
        self._depth_projections += result["depth_count"]
        self._ipm_fallbacks += result["ipm_count"]
        self._confidence_projections += result["confidence_count"]
        self._confidence_missing += result["confidence_missing"]
        self._confidence_rejected += result["confidence_rejected"]
        self._depth_outlier_rejected += result["outlier_rejected"]
        self._last_inference_cameras = {
            "yolo": result["ran_yolo"], "cones": result["ran_cones"]}
        return {"class_grids": class_grids, "observed": observed}

    def _compensate_obstacle_history(self, now):
        if (not self.motion_compensation or self._odom_pose is None
                or not is_fresh(self._odom_stamp, now, self.odom_stale)):
            return
        if (self._filter_pose is not None
                and self._odom_stamp != self._filter_odom_stamp):
            for obstacle_filter in self.obs_filters.values():
                obstacle_filter.compensate_motion(
                    self._filter_pose, self._odom_pose, self.grid)
        self._filter_pose = self._odom_pose
        self._filter_odom_stamp = self._odom_stamp

    # ---- main loop ----
    def _tick(self):
        now = stamp_to_sec(self.get_clock().now().to_msg())
        empty = np.zeros((self.grid.height, self.grid.width), bool)
        road_bev = empty.copy()
        class_grids = {
            k: empty.copy() for k in ("person", "vehicle", "cone", "generic")}
        known = np.zeros((self.grid.height, self.grid.width), bool)
        obstacle_observed = np.zeros_like(known)
        saw_camera = False
        detection_jobs = []

        detection_result = self._consume_detection_result(now)
        if detection_result is not None:
            class_grids = detection_result["class_grids"]
            obstacle_observed |= detection_result["observed"]

        ready_cameras = []
        for cam in self.cameras:
            if (cam.img is not None and is_fresh(cam.stamp, now, self.img_stale)
                    and cam.ensure_homography()):
                ready_cameras.append(cam.name)
        if not self.required_cameras.issubset(ready_cameras):
            return
        selected_yolo = set(self.detection_scheduler.select(
            [name for name in ready_cameras
             if not self.yolo_cams or name in self.yolo_cams]))
        selected_cones = set(self.cone_scheduler.select(
            [name for name in ready_cameras
             if not self.cone_cams or name in self.cone_cams]))
        selected = selected_yolo | selected_cones

        for cam in self.cameras:
            if cam.name not in ready_cameras:
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
            run_yolo = (self.yolo is not None and cam.name in selected_yolo
                        and cam.stamp != cam.last_yolo_stamp)
            run_cones = (self.cones is not None and cam.name in selected_cones
                         and cam.stamp != cam.last_cone_stamp)
            run_classical = (
                cam.name in selected
                and (self.obstacle_method in ("classical", "both")
                     or self.yolo is None)
                and cam.stamp != cam.last_yolo_stamp)
            if self.use_cam_obs and (run_yolo or run_cones or run_classical):
                depth_sample = cam.depth_buffer.nearest(
                    cam.stamp, self.depth_sync)
                confidence_sample = cam.confidence_buffer.nearest(
                    cam.stamp, self.depth_sync)
                waiting_for_zed = (
                    now - cam.stamp < self.depth_wait
                    and (depth_sample is None
                         or (cam.confidence_expected
                             and confidence_sample is None)))
                if waiting_for_zed:
                    self._depth_waits += 1
                    continue
                self._depth_matches += int(depth_sample is not None)
                self._depth_unmatched += int(depth_sample is None)
                detection_jobs.append({
                    "name": cam.name,
                    "image": cam.img,
                    "road": road,
                    "stamp": cam.stamp,
                    "depth": None if depth_sample is None else depth_sample[1],
                    "depth_stamp": 0.0 if depth_sample is None else depth_sample[0],
                    "confidence": (None if confidence_sample is None
                                   else confidence_sample[1]),
                    "K": cam.K,
                    "H": cam.H,
                    "known": cam.known,
                    "cam_xyz": cam.cam_xyz,
                    "pitch": cam.pitch,
                    "yaw": cam.yaw,
                    "run_yolo": run_yolo,
                    "run_cones": run_cones,
                    "run_classical": run_classical,
                    "pose": self._odom_buffer.interpolate(
                        cam.stamp, self.odom_interpolation_tolerance),
                })
                if run_yolo or run_classical:
                    cam.last_yolo_stamp = cam.stamp
                if run_cones:
                    cam.last_cone_stamp = cam.stamp

        if detection_jobs:
            self.detector_worker.submit({
                "cameras": detection_jobs,
            })

        lidar_fresh = (
            self._latest_points is not None
            and is_fresh(self._pts_stamp, now, self.lidar_stale)
        )
        if self.use_lidar and lidar_fresh:
            pts = obstacles.filter_obstacle_points(
                self._latest_points, self.z_min, self.z_max)
            class_grids["generic"] |= obstacles.points_to_grid_mask(pts, self.grid)
            self._publish_obstacle_points(pts)

        # a 360° lidar observes the whole grid, so a fresh lidar frame this
        # tick means "observed" is everything; otherwise it's the union of
        # camera FOVs
        lidar_active = (self.use_lidar and self._latest_points is not None
                        and is_fresh(self._pts_stamp, now, self.lidar_stale))
        if lidar_active:
            obstacle_observed = np.ones_like(obstacle_observed)
        self._compensate_obstacle_history(now)
        if self.temporal_enabled:
            class_grids = {
                group: self.obs_filters[group].update(mask, obstacle_observed)
                for group, mask in class_grids.items()
            }

        obst_grid = np.zeros_like(known)
        for mask in class_grids.values():
            obst_grid |= mask

        if not saw_camera and not lidar_active:  # nothing seen yet
            return
        if self._require_detection_result and not self._have_detection_result:
            return
        if (self._require_detection_result
                and not is_fresh(self._last_detection_result_time, now,
                                 self.detector_stale)):
            return

        layers = {
            group: dict(mask=mask, **self.obstacle_classes[group])
            for group, mask in class_grids.items()
        }

        cost = build_cost_array(
            self.grid, road_bev, obst_grid, known_mask=known,
            offroad_cost=self.offroad_cost,
            inflation_radius=self.inflation_radius,
            cost_scaling_factor=self.cost_scaling_factor,
            unknown_cost=self.unknown_cost,
            obstacle_layers=layers,
            road_edge_radius=self.road_edge_radius,
            road_edge_scaling=self.road_edge_scaling,
            min_offroad_width_m=self.min_offroad_width_m,
            unknown_infill=self.unknown_infill,
            infill_falloff=self.infill_falloff)
        stamp = self.get_clock().now().to_msg()
        msg = to_occupancy_grid_msg(cost, self.grid, stamp=stamp)
        self.costmap_pub.publish(msg)
        self._last_publish_time = now
        self.known_pub.publish(to_occupancy_grid_msg(
            (known.astype(np.int8) * 100), self.grid, stamp=stamp))
        self._ticks += 1
        if self._ticks % 100 == 0:
            self.get_logger().info(
                "accuracy pipeline: yolo=%s cones=%s depth=%d ipm_fallback=%d "
                "sync=%d/%d waits=%d confidence=%d/%d conf_reject=%d "
                "depth_outlier=%d inference=%d/%d/%d" % (
                    sorted(self._last_inference_cameras["yolo"]),
                    sorted(self._last_inference_cameras["cones"]),
                    self._depth_projections,
                    self._ipm_fallbacks,
                    self._depth_matches,
                    self._depth_unmatched,
                    self._depth_waits,
                    self._confidence_projections,
                    self._confidence_missing,
                    self._confidence_rejected,
                    self._depth_outlier_rejected,
                    self.detector_worker.submitted,
                    self.detector_worker.replaced,
                    self.detector_worker.completed))

    def _on_reset(self, request, response):
        """Drop every accumulated observation and start as if freshly launched.

        Clears, in order: per-class temporal confidence, the motion-
        compensation reference pose, buffered lidar/odometry/camera samples,
        and the detection-result gate. Parameters, homographies, loaded models
        and subscriptions are untouched -- this is a memory reset, not a
        restart, so the node is publishing again on the next tick.

        Nothing is persisted to disk anywhere in this pipeline, and both Nav2
        costmaps are rolling with no static layer, so after this call and a
        Nav2 costmap clear the vehicle genuinely holds no prior-run state.
        """
        cleared = sum(f.reset() for f in self.obs_filters.values())

        self._filter_pose = None
        self._filter_odom_stamp = None
        self._latest_points = None
        self._pts_stamp = None
        self._odom_pose = None
        self._odom_stamp = None
        clear_sample_buffer(self._odom_buffer)

        for cam in self.cameras:
            cam.img, cam.stamp = None, 0.0
            cam.last_yolo_stamp = None
            cam.last_cone_stamp = None
            clear_sample_buffer(cam.depth_buffer)
            clear_sample_buffer(cam.confidence_buffer)

        self._have_detection_result = False
        self._last_detection_result_time = None
        self._last_publish_time = None
        self._last_inference_cameras = {"yolo": [], "cones": []}
        for counter in ("_depth_projections", "_ipm_fallbacks",
                        "_confidence_projections", "_confidence_missing",
                        "_confidence_rejected", "_depth_outlier_rejected",
                        "_depth_matches", "_depth_unmatched", "_depth_waits",
                        "_ticks", "_detector_errors"):
            setattr(self, counter, 0)
        self._last_detector_error_time = None

        response.success = True
        response.message = (
            "perception reset: %d lethal cells cleared across %d temporal "
            "filters; no prior-run state retained"
            % (cleared, len(self.obs_filters)))
        self.get_logger().warning(response.message)
        return response

    def destroy_node(self):
        if hasattr(self, "detector_worker"):
            self.detector_worker.close(timeout=5.0)
        return super().destroy_node()

    @staticmethod
    def _buffer_age(buffer, now):
        if not buffer.samples:
            return None
        return max(0.0, now - buffer.samples[-1][0])

    def _publish_diagnostics(self):
        now = stamp_to_sec(self.get_clock().now().to_msg())
        report = DiagnosticArray()
        report.header.stamp = self.get_clock().now().to_msg()
        statuses = []
        for camera in self.cameras:
            image_age = None if camera.img is None else max(0.0, now - camera.stamp)
            depth_age = self._buffer_age(camera.depth_buffer, now)
            confidence_age = self._buffer_age(camera.confidence_buffer, now)
            level, message = camera_health(
                image_age, depth_age, confidence_age, self.img_stale,
                camera.confidence_expected)
            status = DiagnosticStatus(
                level=bytes([level]),
                name="perception_costmap/camera/%s" % camera.name,
                hardware_id="zed_%s" % camera.name,
                message=message)
            status.values = [
                KeyValue(key="rgb_age_sec", value=str(image_age)),
                KeyValue(key="depth_age_sec", value=str(depth_age)),
                KeyValue(key="confidence_age_sec", value=str(confidence_age)),
            ]
            statuses.append(status)

        detector_age = (None if self._last_detection_result_time is None
                        else max(0.0, now - self._last_detection_result_time))
        detector_ok = (detector_age is not None
                       and detector_age <= self.detector_stale
                       and (self._last_detector_error_time is None
                            or self._last_detection_result_time
                            > self._last_detector_error_time))
        detector = DiagnosticStatus(
            level=bytes([0 if detector_ok else 2]),
            name="perception_costmap/detectors",
            hardware_id="jetson_tensorrt",
            message="detectors current" if detector_ok else "detectors stale or failed")
        detector.values = [
            KeyValue(key="result_age_sec", value=str(detector_age)),
            KeyValue(key="errors", value=str(self._detector_errors)),
            KeyValue(key="submitted", value=str(self.detector_worker.submitted)),
            KeyValue(key="replaced", value=str(self.detector_worker.replaced)),
            KeyValue(key="completed", value=str(self.detector_worker.completed)),
        ]
        statuses.append(detector)

        publish_age = (None if self._last_publish_time is None
                       else max(0.0, now - self._last_publish_time))
        output_ok = publish_age is not None and publish_age <= 0.25
        statuses.append(DiagnosticStatus(
            level=bytes([0 if output_ok else 2]),
            name="perception_costmap/output",
            hardware_id="perception_costmap",
            message="costmap current" if output_ok else "costmap publication stale",
            values=[KeyValue(key="costmap_age_sec", value=str(publish_age))]))
        report.status = statuses
        self.health_pub.publish(report)

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
