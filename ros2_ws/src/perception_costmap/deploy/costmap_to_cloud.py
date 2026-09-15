#!/usr/bin/env python3
"""Publish /perception/costmap's obstacle cells as a PointCloud2 for Nav2's
ObstacleLayer.

Why not StaticLayer (two failed attempts before this):
  1. Fed the raw base_link grid -> StaticLayer places a robot-relative map
     once and never re-places it as the robot moves.
  2. Fed a grid whose origin was transformed into odom -> Costmap2D is
     axis-aligned and silently DISCARDS the origin's yaw, so every obstacle
     landed rotated by the robot's heading. Measured: perception had the
     obstacle at 1.37 m dead ahead while nav2 read cost=0 there.
  3. Resampling to an axis-aligned odom grid fixes the rotation, but the grid
     then changes size/origin every frame -- a StaticLayer is built for a
     FIXED map and thrashes on that.

ObstacleLayer is the layer meant for this: it consumes sensor observations in
a sensor frame, transforms them through TF every update, marks obstacles and
raytrace-clears free space. No rotation assumption, no fixed-map assumption.
So we hand the costmap's occupied cells over as a point cloud in base_link
and let ObstacleLayer do what it is designed to do.
"""
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from perception_costmap.costmap_cloud import raycast_costmap

class CostmapToCloud(Node):
    def __init__(self):
        super().__init__("costmap_to_cloud")
        self.declare_parameter("obstacle_threshold", 97)
        self.declare_parameter("point_z", 0.35)
        self.declare_parameter("angle_min_deg", -100.0)
        self.declare_parameter("angle_max_deg", 100.0)
        self.declare_parameter("angle_increment_deg", 0.5)
        self.declare_parameter("min_range_m", 0.3)
        self.declare_parameter("obstacle_range_m", 15.0)
        self.declare_parameter("raytrace_range_m", 16.0)

        self.obstacle_threshold = int(
            self.get_parameter("obstacle_threshold").value)
        self.point_z = float(self.get_parameter("point_z").value)
        angle_min = float(self.get_parameter("angle_min_deg").value)
        angle_max = float(self.get_parameter("angle_max_deg").value)
        angle_step = float(self.get_parameter("angle_increment_deg").value)
        self.min_range = float(self.get_parameter("min_range_m").value)
        self.obstacle_range = float(
            self.get_parameter("obstacle_range_m").value)
        self.raytrace_range = float(
            self.get_parameter("raytrace_range_m").value)
        if not (0.0 < self.min_range < self.obstacle_range < self.raytrace_range):
            raise ValueError(
                "expected min_range < obstacle_range < raytrace_range")
        if angle_step <= 0.0 or angle_max <= angle_min:
            raise ValueError("invalid angular sampling configuration")

        self.angles = np.radians(
            np.arange(angle_min, angle_max + 0.5 * angle_step, angle_step))
        self.ranges = np.arange(self.min_range, self.obstacle_range, 0.05)
        self.ray_x = np.cos(self.angles)[:, None] * self.ranges[None, :]
        self.ray_y = np.sin(self.angles)[:, None] * self.ranges[None, :]
        self.clear_x = np.cos(self.angles) * self.raytrace_range
        self.clear_y = np.sin(self.angles) * self.raytrace_range

        self.pub = self.create_publisher(PointCloud2, "/perception/costmap_cloud", 1)
        self.create_subscription(OccupancyGrid, "/perception/costmap",
                                 self.cb, qos_profile_sensor_data)
        self.n = 0

    def cb(self, msg):
        r = msg.info.resolution
        W, H = msg.info.width, msg.info.height
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y
        grid = np.asarray(msg.data, dtype=np.int8).reshape(H, W)

        # Emit only the FIRST obstacle cell along each ray from the robot.
        #
        # The perception costmap marks the occlusion shadow BEHIND an obstacle
        # as lethal (100) -- identical to the obstacle itself. Measured with a
        # dummy at 5 m: 7523 lethal cells forming a solid wall from 5 m out to
        # 16 m spanning the whole lane, so any goal past it was unreachable and
        # the planner correctly returned an empty path. Dumping every occupied
        # cell into the cloud reproduces that wall inside nav2.
        #
        # Ray-casting and keeping only the nearest hit per bearing yields the
        # visible surface of obstacles. Space behind them is simply never
        # marked, so it stays free/unknown and the planner can route around --
        # which is the whole point. Anything genuinely there gets marked as
        # soon as the robot moves and actually observes it.
        # Coverage: +/-100 deg bearing, 0.3-15 m range. The grid extends to
        # 16 m forward / +/-10 m lateral, so obstacles beyond 15 m or in the
        # far rear-lateral corners are not emitted -- acceptable at the 1-2 mph
        # cap, but widen these if the speed cap or grid extent changes.
        # Every bearing gets an endpoint. Hits end on the nearest obstacle;
        # clear bearings end beyond obstacle_max_range so Nav2 raytraces them
        # without marking a synthetic obstacle at the endpoint.
        pts, has_hit = raycast_costmap(
            grid, r, ox, oy, self.ray_x, self.ray_y,
            self.clear_x, self.clear_y, self.obstacle_threshold, self.point_z)
        rows = np.flatnonzero(has_hit)

        header = msg.header          # base_link, same stamp -- TF handles the rest
        cloud = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.pub.publish(cloud)

        self.n += 1
        # Cheap, infrequent guard against the offroad_cost/obstacle_threshold
        # coupling silently breaking road-keeping. Once after ~5 s, then hourly-ish.
        if self.n == 50 or self.n % 600 == 0:
            self._warn_if_offroad_below_threshold(grid)
        if self.n % 20 == 0:
            self.get_logger().info(
                f"published {self.n} clouds, latest {len(rows)} marked rays, "
                f"{len(pts) - len(rows)} clearing rays "
                f"(frame={header.frame_id})")


    def _warn_if_offroad_below_threshold(self, grid):
        """Warn when the source grid's off-road plateau sits below our threshold.

        Only cells at or above ``obstacle_threshold`` are forwarded to Nav2.
        Obstacles are always LETHAL (100) so they always get through, which is
        exactly what makes a mismatch so quiet: avoidance keeps working while
        road-keeping is gone. perception_costmap.yaml shipped for a long time
        without setting offroad_cost at all, defaulting it to 65 -- below the
        97 assumed here.

        Off-road is one constant painted over a large area, so it shows up as a
        single sub-threshold value covering a big fraction of the grid. That is
        the signature we look for; a scene that genuinely has no off-road in it
        produces no such plateau and no warning.
        """
        below = grid[(grid > 0) & (grid < self.obstacle_threshold)]
        if below.size == 0:
            return
        counts = np.bincount(below.astype(np.int64),
                             minlength=self.obstacle_threshold + 1)
        value = int(counts.argmax())
        fraction = float(counts[value]) / float(grid.size)
        if fraction < 0.10:
            return
        self.get_logger().warning(
            "%.0f%% of /perception/costmap sits at cost %d, below this "
            "bridge's obstacle_threshold of %d. Cells below the threshold are "
            "NOT forwarded to Nav2, so if %d is the perception node's "
            "offroad_cost, ROAD-KEEPING IS DISABLED -- Nav2 will avoid "
            "obstacles but not the road edge. Set offroad_cost >= %d in the "
            "perception config (perception_dinosaur.yaml uses 97), or lower "
            "obstacle_threshold here to match."
            % (fraction * 100.0, value, self.obstacle_threshold, value,
               self.obstacle_threshold))


def main():
    rclpy.init()
    node = CostmapToCloud()
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
