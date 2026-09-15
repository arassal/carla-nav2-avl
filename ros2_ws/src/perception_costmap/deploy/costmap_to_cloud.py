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
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
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
        # Node whose offroad_cost must reach obstacle_threshold (see below).
        self.declare_parameter("perception_node", "perception_costmap")

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

        # Road-keeping only works if the perception node paints off-road at or
        # above obstacle_threshold: cells below it are never forwarded. Ask that
        # node for its real offroad_cost instead of guessing from the grid --
        # a large sub-threshold plateau can just as well be unknown_cost (25 on
        # the car), which is what the earlier heuristic mistook for off-road.
        node_name = str(self.get_parameter("perception_node").value).strip("/")
        self._offroad_client = self.create_client(
            GetParameters, "/%s/get_parameters" % node_name)
        self._offroad_pending = False
        self._offroad_seen = None
        self.create_timer(10.0, self._check_offroad_cost)

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
        if self.n % 20 == 0:
            self.get_logger().info(
                f"published {self.n} clouds, latest {len(rows)} marked rays, "
                f"{len(pts) - len(rows)} clearing rays "
                f"(frame={header.frame_id})")


    def _check_offroad_cost(self):
        """Re-read the perception node's offroad_cost; log only when it changes.

        Rechecked every 10 s so a restarted perception node with a different
        config is noticed. Silent while that node isn't up.
        """
        if self._offroad_pending or not self._offroad_client.service_is_ready():
            return
        self._offroad_pending = True
        future = self._offroad_client.call_async(
            GetParameters.Request(names=["offroad_cost"]))
        future.add_done_callback(self._on_offroad_cost)

    def _on_offroad_cost(self, future):
        self._offroad_pending = False
        try:
            values = future.result().values
        except Exception:  # node went away mid-call; the timer retries
            return
        if not values or values[0].type != ParameterType.PARAMETER_INTEGER:
            return
        cost = int(values[0].integer_value)
        if cost == self._offroad_seen:
            return
        self._offroad_seen = cost
        if cost < self.obstacle_threshold:
            self.get_logger().warning(
                "perception offroad_cost is %d, below this bridge's "
                "obstacle_threshold of %d: off-road cells are NOT forwarded to "
                "Nav2, so ROAD-KEEPING IS DISABLED -- Nav2 will avoid obstacles "
                "but not the road edge. Set offroad_cost >= %d in the perception "
                "config (perception_dinosaur.yaml uses 97), or lower "
                "obstacle_threshold here to match."
                % (cost, self.obstacle_threshold, self.obstacle_threshold))
        else:
            self.get_logger().info(
                "perception offroad_cost %d >= obstacle_threshold %d: road edges "
                "reach Nav2" % (cost, self.obstacle_threshold))


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
