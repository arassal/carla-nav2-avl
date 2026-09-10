#!/usr/bin/env python3
"""Publish the campus road graph as RViz markers -- decimated for speed.

RViz on this machine runs under llvmpipe (software GL): the NoMachine virtual
display exposes no usable hardware GL context, so every vertex is rasterised
on the CPU. The full graph is 16,651 nodes + 17,492 edges, and drawing all of
it pushed RViz past 100% CPU on its own.

Two decimations, both configurable:

  publish_nodes (default False)
      The 16,651 node POINTS are pure decoration -- the edges already show
      where the roads are. Dropping them removes ~16k primitives outright.

  max_range_m (default 300.0, 0 disables)
      Only draw edges within this radius of the robot. A drive is typically
      ~100 m, so 300 m is generous while cutting a campus-wide graph down to
      the neighbourhood you are actually choosing a destination in. The
      marker is republished as the robot moves so the window follows it.

Set max_range_m:=0.0 to draw the whole campus (slower, but useful if you want
to click somewhere far away).
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


class RouteGraphViz(Node):
    def __init__(self):
        super().__init__('route_graph_viz')

        self.declare_parameter('graph_file', '')
        self.declare_parameter('frame', 'map')
        self.declare_parameter('publish_nodes', False)
        self.declare_parameter('max_range_m', 300.0)
        self.declare_parameter('republish_period_s', 3.0)
        self.declare_parameter('line_width', 0.6)

        self.graph_file = self.get_parameter('graph_file').value
        self.frame = self.get_parameter('frame').value
        self.publish_nodes = self.get_parameter('publish_nodes').value
        self.max_range = float(self.get_parameter('max_range_m').value)
        self.line_width = float(self.get_parameter('line_width').value)

        if not self.graph_file:
            self.get_logger().error('graph_file parameter is required')
            raise SystemExit(1)

        with open(self.graph_file) as f:
            data = json.load(f)

        # Flatten once at startup so the periodic republish is cheap.
        self.segments = []      # [(x1, y1, x2, y2), ...]
        self.nodes = []         # [(x, y), ...]
        for feat in data.get('features', []):
            geom = feat.get('geometry', {})
            gtype = geom.get('type')
            coords = geom.get('coordinates')
            if gtype == 'Point':
                self.nodes.append((float(coords[0]), float(coords[1])))
            elif gtype in ('LineString', 'MultiLineString'):
                segs = [coords] if gtype == 'LineString' else coords
                for seg in segs:
                    for a, b in zip(seg[:-1], seg[1:]):
                        self.segments.append((float(a[0]), float(a[1]),
                                              float(b[0]), float(b[1])))

        self.robot = None
        self.create_subscription(Odometry, '/odometry/global', self._on_odom, 10)

        qos = QoSProfile(depth=1,
                         durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(MarkerArray, '/route_graph', qos)

        self._last_center = None
        self.create_timer(float(self.get_parameter('republish_period_s').value),
                          self._tick)
        self._tick()   # draw something immediately, before any odom arrives

        self.get_logger().info(
            f'route graph loaded: {len(self.nodes)} nodes, {len(self.segments)} '
            f'segments | publish_nodes={self.publish_nodes} '
            f'max_range_m={self.max_range}')

    def _on_odom(self, msg):
        self.robot = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _tick(self):
        cx, cy = self.robot if self.robot else (0.0, 0.0)

        # Only redraw when the robot has actually moved a meaningful distance;
        # re-serialising 35k vertices every tick would defeat the point.
        if self._last_center is not None and self.max_range > 0:
            dx = cx - self._last_center[0]
            dy = cy - self._last_center[1]
            if math.hypot(dx, dy) < 10.0:
                return
        self._last_center = (cx, cy)

        edges = Marker()
        edges.header.frame_id = self.frame
        edges.ns = 'edges'
        edges.id = 0
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.scale.x = self.line_width
        edges.color.r, edges.color.g, edges.color.b, edges.color.a = 0.2, 0.7, 1.0, 0.85
        edges.pose.orientation.w = 1.0

        r2 = self.max_range ** 2
        for x1, y1, x2, y2 in self.segments:
            if self.max_range > 0:
                # keep the segment if either end is in range
                if ((x1 - cx) ** 2 + (y1 - cy) ** 2 > r2 and
                        (x2 - cx) ** 2 + (y2 - cy) ** 2 > r2):
                    continue
            pa, pb = Point(), Point()
            pa.x, pa.y = x1, y1
            pb.x, pb.y = x2, y2
            edges.points.append(pa)
            edges.points.append(pb)

        markers = [edges]

        if self.publish_nodes:
            nodes = Marker()
            nodes.header.frame_id = self.frame
            nodes.ns = 'nodes'
            nodes.id = 1
            nodes.type = Marker.POINTS
            nodes.action = Marker.ADD
            nodes.scale.x = nodes.scale.y = 1.2
            nodes.color.r, nodes.color.g, nodes.color.b, nodes.color.a = 1.0, 0.75, 0.1, 0.9
            nodes.pose.orientation.w = 1.0
            for x, y in self.nodes:
                if self.max_range > 0 and (x - cx) ** 2 + (y - cy) ** 2 > r2:
                    continue
                p = Point()
                p.x, p.y = x, y
                nodes.points.append(p)
            markers.append(nodes)

        arr = MarkerArray()
        arr.markers = markers
        self.pub.publish(arr)
        self.get_logger().debug(f'republished graph: {len(edges.points)} verts')


def main():
    rclpy.init()
    try:
        node = RouteGraphViz()
    except SystemExit:
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
