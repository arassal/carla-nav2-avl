"""
Rolling costmap node with pedestrian-aware inflation.

Subscribes:
  /scan      — vehicles, buildings, static obstacles  → 2.0 m inflation
  /ped_scan  — pedestrians only                       → 5.0 m inflation

The car's real footprint (4.7 m × 1.86 m) sets the minimum safe clearance.
Pedestrians get a much larger danger zone so the car stays well clear of people
compared to parked cars or walls.

Publishes:
  /local_costmap/costmap   (OccupancyGrid, 40 m × 40 m, 0.25 m/cell, 5 Hz)
  /global_costmap/costmap  (OccupancyGrid, 150 m × 150 m, 0.5 m/cell, 1 Hz)
"""

import math
import numpy as np
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion
import tf2_ros

# ── grid parameters ───────────────────────────────────────────────────────────

LOCAL_W     = 40.0
LOCAL_RES   = 0.25
LOCAL_CELLS = int(LOCAL_W / LOCAL_RES)   # 160

GLOBAL_W     = 150.0
GLOBAL_RES   = 0.5
GLOBAL_CELLS = int(GLOBAL_W / GLOBAL_RES)  # 300

# Tesla Model 3 half-width = 0.93 m → minimum obstacle clearance from car centre.
# OBS_INFLATION:  car body (0.93) + road margin (1.07)  = 2.0 m
# PED_INFLATION:  car body (0.93) + pedestrian buffer (4.07) = 5.0 m
OBS_INFLATION_M = 2.0
PED_INFLATION_M = 5.0

LETHAL    = 100   # occupancy value for an actual obstacle cell
NEAR_LETHAL = 99  # max value for inflated (danger zone) cells

LOCAL_PUB_HZ  = 5.0
GLOBAL_PUB_HZ = 1.0
DECAY_HZ      = 0.25


# ── kernel builder ────────────────────────────────────────────────────────────

def _build_kernel(radius_m: float, resolution: float) -> np.ndarray:
    r    = int(math.ceil(radius_m / resolution))
    size = 2 * r + 1
    k    = np.zeros((size, size), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            d = math.hypot(x - r, y - r) * resolution
            if d <= radius_m:
                k[y, x] = NEAR_LETHAL * max(0.0, 1.0 - d / radius_m)
    return k


def _inflate(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.ndimage import maximum_filter
    lethal   = (grid == LETHAL).astype(np.float32) * NEAR_LETHAL
    inflated = maximum_filter(lethal, footprint=(kernel > 0))
    result   = np.where(grid == LETHAL, LETHAL,
                        np.clip(inflated, 0, NEAR_LETHAL).astype(np.int8))
    return result.astype(np.int8)


# ── node ──────────────────────────────────────────────────────────────────────

class CostmapNode(Node):
    def __init__(self):
        super().__init__('costmap_node')
        self._lock = threading.Lock()

        # Obstacle grids (regular obstacles)
        self._local_obs  = np.zeros((LOCAL_CELLS,  LOCAL_CELLS),  dtype=np.int8)
        self._global_obs = np.zeros((GLOBAL_CELLS, GLOBAL_CELLS), dtype=np.int8)

        # Pedestrian grids (separate layer, larger inflation at publish time)
        self._local_ped  = np.zeros((LOCAL_CELLS,  LOCAL_CELLS),  dtype=np.int8)
        self._global_ped = np.zeros((GLOBAL_CELLS, GLOBAL_CELLS), dtype=np.int8)

        # Pre-built kernels
        self._k_local_obs  = _build_kernel(OBS_INFLATION_M,        LOCAL_RES)
        self._k_local_ped  = _build_kernel(PED_INFLATION_M,        LOCAL_RES)
        self._k_global_obs = _build_kernel(OBS_INFLATION_M * 1.25, GLOBAL_RES)
        self._k_global_ped = _build_kernel(PED_INFLATION_M * 1.25, GLOBAL_RES)

        self._rx = 0.0
        self._ry = 0.0

        # TF
        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        # QoS
        tqos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.create_subscription(LaserScan, '/scan',     self._on_scan,     10)
        self.create_subscription(LaserScan, '/ped_scan', self._on_ped_scan, 10)

        # Publishers
        self._pub_local  = self.create_publisher(OccupancyGrid,
                                                  '/local_costmap/costmap',  tqos)
        self._pub_global = self.create_publisher(OccupancyGrid,
                                                  '/global_costmap/costmap', tqos)

        self.create_timer(1.0 / LOCAL_PUB_HZ,  self._publish_local)
        self.create_timer(1.0 / GLOBAL_PUB_HZ, self._publish_global)
        self.create_timer(1.0 / DECAY_HZ,       self._decay_global)

        self.get_logger().info(
            f'Costmap started — obs inflation {OBS_INFLATION_M} m, '
            f'pedestrian inflation {PED_INFLATION_M} m')

    # ── TF helpers ────────────────────────────────────────────────────────────

    def _robot_pose(self):
        try:
            t = self._tf_buf.lookup_transform(
                'odom', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0))
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

    def _scan_to_odom(self, msg: LaserScan):
        src_frame = msg.header.frame_id or 'base_link'
        try:
            t = self._tf_buf.lookup_transform(
                'odom', src_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0))
        except Exception:
            return None

        tx  = t.transform.translation.x
        ty  = t.transform.translation.y
        q   = t.transform.rotation
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < msg.range_max)
        if not valid.any():
            return None

        lx = ranges[valid] * np.cos(angles[valid])
        ly = ranges[valid] * np.sin(angles[valid])
        cy, sy = math.cos(yaw), math.sin(yaw)
        ox = tx + cy * lx - sy * ly
        oy = ty + sy * lx + cy * ly
        return np.stack([ox, oy], axis=1)

    # ── grid marking helpers ──────────────────────────────────────────────────

    def _mark(self, grid, pts, cells, resolution, clear=False):
        rx, ry   = self._rx, self._ry
        half     = cells * resolution / 2.0
        ox, oy   = rx - half, ry - half
        if clear:
            grid[:] = 0
        xs = ((pts[:, 0] - ox) / resolution).astype(int)
        ys = ((pts[:, 1] - oy) / resolution).astype(int)
        m  = (xs >= 0) & (xs < cells) & (ys >= 0) & (ys < cells)
        grid[ys[m], xs[m]] = LETHAL

    # ── scan callbacks ────────────────────────────────────────────────────────

    def _on_scan(self, msg: LaserScan):
        pose = self._robot_pose()
        if pose is None:
            return
        pts = self._scan_to_odom(msg)
        if pts is None:
            return
        with self._lock:
            self._rx, self._ry = pose
            self._mark(self._local_obs,  pts, LOCAL_CELLS,  LOCAL_RES,  clear=True)
            self._mark(self._global_obs, pts, GLOBAL_CELLS, GLOBAL_RES, clear=False)

    def _on_ped_scan(self, msg: LaserScan):
        pose = self._robot_pose()
        if pose is None:
            return
        pts = self._scan_to_odom(msg)
        if pts is None:
            return
        with self._lock:
            self._rx, self._ry = pose
            self._mark(self._local_ped,  pts, LOCAL_CELLS,  LOCAL_RES,  clear=True)
            self._mark(self._global_ped, pts, GLOBAL_CELLS, GLOBAL_RES, clear=False)

    # ── grid message builder ──────────────────────────────────────────────────

    def _make_msg(self, grid: np.ndarray, resolution: float,
                  frame: str = 'odom') -> OccupancyGrid:
        rx, ry = self._rx, self._ry
        h, w   = grid.shape
        msg = OccupancyGrid()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.info.resolution = resolution
        msg.info.width      = w
        msg.info.height     = h
        msg.info.origin     = Pose(
            position=Point(x=rx - w * resolution / 2.0,
                           y=ry - h * resolution / 2.0,
                           z=0.0),
            orientation=Quaternion(w=1.0))
        msg.data = grid.flatten().tolist()
        return msg

    # ── publish timers ────────────────────────────────────────────────────────

    def _publish_local(self):
        with self._lock:
            # Inflate each layer separately, then take the highest cost per cell
            obs_layer = _inflate(self._local_obs, self._k_local_obs)
            ped_layer = _inflate(self._local_ped, self._k_local_ped)
            merged    = np.maximum(obs_layer, ped_layer)
            # Restore lethal values that might have been clipped
            merged[self._local_obs == LETHAL] = LETHAL
            merged[self._local_ped == LETHAL] = LETHAL
            msg = self._make_msg(merged, LOCAL_RES)
        self._pub_local.publish(msg)

    def _publish_global(self):
        with self._lock:
            obs_layer = _inflate(self._global_obs, self._k_global_obs)
            ped_layer = _inflate(self._global_ped, self._k_global_ped)
            merged    = np.maximum(obs_layer, ped_layer)
            merged[self._global_obs == LETHAL] = LETHAL
            merged[self._global_ped == LETHAL] = LETHAL
            msg = self._make_msg(merged, GLOBAL_RES, frame='map')
        self._pub_global.publish(msg)

    def _decay_global(self):
        with self._lock:
            for grid in (self._global_obs, self._global_ped):
                lethal = grid == LETHAL
                grid[lethal] = max(0, LETHAL - 5)
                near  = (grid > 50) & ~lethal
                grid[near] = np.maximum(0, grid[near].astype(int) - 2).astype(np.int8)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
