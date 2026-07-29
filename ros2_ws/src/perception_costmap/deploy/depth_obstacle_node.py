#!/usr/bin/env python3
"""Standalone camera-depth obstacle detector.

Independent from the main costmap_node and from lidar by design (per
operator decision 2026-07-16: prove this works on its own before ever
combining it with anything else). Subscribes to the 3 ZED X point clouds,
converts each to a robot-frame obstacle mask, and publishes the union on its
own topics -- costmap_node does NOT currently read these; nothing here
changes the live perception/costmap pipeline.

Runs as a separate process specifically so its cost (measured in isolation:
~4ms/tick front, ~40ms/tick each side camera, occasional spikes to ~100ms)
can never block the main ~8Hz costmap tick. Each camera callback runs
independently; a periodic timer publishes the best available combined
picture rather than waiting for all 3 in lockstep, so one slow/stale camera
doesn't stall the others.

TRANSFORM: reuses bev.homography_from_extrinsics' own rotation construction
(the SAME pitch/yaw/xyz already calibrated for BEV projection), NOT the
static URDF/TF mount angles. Verified live (2026-07-16) that TF disagrees
with the calibrated extrinsics -- e.g. left camera TF says pitch=0 deg,
yaw=90 deg, but perception_dinosaur.yaml's person-walk-calibrated values
are pitch=18 deg, yaw=94 deg -- and using TF's stale angles produced a
physically implausible result (side cameras keeping ~90% of points in the
obstacle height band vs front's ~50%; correct extrinsics gave a consistent
35-46% across all three).

The ZED SDK point cloud is in camera-BODY convention (x-forward along
boresight, y-left, z-up), not optical convention, so bev.py's fixed R0
(body<->optical axis swap) converts it before applying bev.py's own
world<->optical rotation R, inverted:
    point_robot = R.T @ (R0 @ point_body) + cam_xyz
"""
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

from perception_costmap import obstacles
from perception_costmap.occupancy import GridSpec, to_occupancy_grid_msg

R0 = np.array([[0.0, -1.0, 0.0],
              [0.0, 0.0, -1.0],
              [1.0, 0.0, 0.0]])


def calibrated_rotation(pitch_deg, yaw_deg):
    th = np.radians(pitch_deg)
    yw = np.radians(yaw_deg)
    Rx = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.cos(th), -np.sin(th)],
                  [0.0, np.sin(th), np.cos(th)]])
    Rz_inv = np.array([[np.cos(yw), np.sin(yw), 0.0],
                       [-np.sin(yw), np.cos(yw), 0.0],
                       [0.0, 0.0, 1.0]])
    return Rx @ R0 @ Rz_inv


class DepthObstacleNode(Node):
    def __init__(self):
        super().__init__('depth_obstacle_node')

        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('x_min', -4.0)
        self.declare_parameter('x_max', 16.0)
        self.declare_parameter('y_min', -10.0)
        self.declare_parameter('y_max', 10.0)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('z_min', 0.2)
        self.declare_parameter('z_max', 2.0)
        self.declare_parameter('stale_sec', 1.0)
        self.declare_parameter('publish_rate', 5.0)
        # Isolation as a separate process did NOT prevent it from slowing the
        # main costmap tick (measured live: 5.85 -> 2.8 Hz with this node
        # running) -- the Jetson's shared memory bandwidth is the likely
        # bottleneck, not CPU core count (12 cores, plenty nominally free).
        # Stride-downsample before any processing to cut the real cost, not
        # just move it to another process.
        self.declare_parameter('point_stride', 8)
        self.declare_parameter('cameras', ['front', 'left', 'right'])
        # Calibrated extrinsics -- same numbers as perception_dinosaur.yaml,
        # NOT the URDF/TF mount angles (see module docstring for why).
        self.declare_parameter('front.xyz', [0.6795, 0.0, 0.4476])
        self.declare_parameter('front.pitch_deg', 17.0)
        self.declare_parameter('front.yaw_deg', 0.0)
        self.declare_parameter('left.xyz', [0.098, 0.286, 0.6126])
        self.declare_parameter('left.pitch_deg', 18.0)
        self.declare_parameter('left.yaw_deg', 94.0)
        self.declare_parameter('right.xyz', [0.098, -0.286, 0.6126])
        self.declare_parameter('right.pitch_deg', 20.0)
        self.declare_parameter('right.yaw_deg', -86.0)

        g = lambda n: self.get_parameter(n).value
        self.grid = GridSpec(x_min=g('x_min'), x_max=g('x_max'),
                             y_min=g('y_min'), y_max=g('y_max'),
                             resolution=g('resolution'), frame_id=g('frame_id'))
        self.z_min, self.z_max = g('z_min'), g('z_max')
        self.stale_sec = g('stale_sec')
        self.point_stride = g('point_stride')
        self.cams = list(g('cameras'))

        self.xforms = {}
        self.last_mask = {}
        self.last_t = {}
        self.msg_count = {}
        for c in self.cams:
            xyz = g(f'{c}.xyz')
            R = calibrated_rotation(g(f'{c}.pitch_deg'), g(f'{c}.yaw_deg'))
            self.xforms[c] = (R.T @ R0, np.array(xyz))
            self.last_mask[c] = None
            self.last_t[c] = 0.0
            self.msg_count[c] = 0
            topic = f'/zed_{c}/zed_node/point_cloud/cloud_registered'
            self.create_subscription(PointCloud2, topic, self._make_cb(c),
                                     qos_profile_sensor_data)

        self.grid_pub = self.create_publisher(OccupancyGrid, '/perception/depth_obstacle_grid', 1)
        self.points_pub = self.create_publisher(PointCloud2, '/perception/depth_obstacle_points', 1)
        self.create_timer(1.0 / g('publish_rate'), self._publish_combined)
        self.get_logger().info(
            f'depth_obstacle_node up: cameras={self.cams}, '
            f'z_band=[{self.z_min},{self.z_max}]m, point_stride={self.point_stride}, '
            f'independent of lidar/costmap_node')

    def _make_cb(self, cam):
        from sensor_msgs_py import point_cloud2
        M, t = self.xforms[cam]

        def cb(msg):
            pts_cam = point_cloud2.read_points_numpy(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)
            if self.point_stride > 1:
                pts_cam = pts_cam[::self.point_stride]
            finite = np.isfinite(pts_cam).all(axis=1)
            pts_cam = pts_cam[finite]
            if pts_cam.shape[0] == 0:
                return
            pts_robot = pts_cam @ M.T + t
            filtered = obstacles.filter_obstacle_points(pts_robot, self.z_min, self.z_max)
            mask = obstacles.points_to_grid_mask(filtered, self.grid)

            self.last_mask[cam] = mask
            self.last_t[cam] = time.monotonic()
            self.msg_count[cam] += 1
        return cb

    def _publish_combined(self):
        now = time.monotonic()
        shape = (self.grid.height, self.grid.width)
        combined = np.zeros(shape, dtype=bool)
        any_fresh = False
        for c in self.cams:
            if self.last_mask[c] is None:
                continue
            if now - self.last_t[c] > self.stale_sec:
                continue  # camera gone stale -- drop it, don't smear old data
            combined |= self.last_mask[c]
            any_fresh = True
        if not any_fresh:
            return

        stamp = self.get_clock().now().to_msg()
        cost = np.where(combined, 100, 0).astype(np.int8)
        self.grid_pub.publish(to_occupancy_grid_msg(cost, self.grid, stamp=stamp))
        self._publish_points(combined, stamp)

    def _publish_points(self, mask, stamp):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return
        wx = self.grid.x_min + (xs + 0.5) * self.grid.resolution
        wy = self.grid.y_min + (ys + 0.5) * self.grid.resolution
        wz = np.zeros_like(wx)
        pts = np.column_stack((wx, wy, wz)).astype(np.float32)

        msg = PointCloud2()
        msg.header = Header(stamp=stamp, frame_id=self.grid.frame_id)
        msg.height = 1
        msg.width = pts.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.data = pts.tobytes()
        self.points_pub.publish(msg)


def main():
    rclpy.init()
    node = DepthObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
