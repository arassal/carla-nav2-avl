#!/usr/bin/env python3
"""Publish the nav2_route GeoJSON graph as RViz markers.

Visualization only: the graph coordinates are map-frame meters relative to
the datum baked in by generate_graph.py. Nothing localizes us against it
(no map->odom localizer runs on this vehicle), so a separate static
map->odom identity TF is needed for the graph to render alongside live
sensor data -- the robot's apparent position on the graph is therefore
wherever odom happened to start, NOT a real global fix.
"""
import json
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import os
# Route graph geojson. Override with ROUTE_GRAPH env var; the default is the
# path on our car and will not exist on a fresh clone.
GRAPH = os.environ.get(
    'ROUTE_GRAPH',
    '/home/dinosaur/IGVC/install/avros_bringup/share/avros_bringup/config/cpp_campus_graph.geojson')
FRAME = 'map'


class RouteGraphViz(Node):
    def __init__(self):
        super().__init__('route_graph_viz')

        # Latched: RViz subscribes long after we publish once.
        qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.pub = self.create_publisher(MarkerArray, '/route_graph', qos)

        with open(GRAPH) as f:
            data = json.load(f)

        edges = Marker()
        edges.header.frame_id = FRAME
        edges.ns = 'edges'
        edges.id = 0
        edges.type = Marker.LINE_LIST
        edges.action = Marker.ADD
        edges.scale.x = 0.7
        edges.color.r, edges.color.g, edges.color.b, edges.color.a = 0.2, 0.7, 1.0, 0.85
        edges.pose.orientation.w = 1.0

        nodes = Marker()
        nodes.header.frame_id = FRAME
        nodes.ns = 'nodes'
        nodes.id = 1
        nodes.type = Marker.POINTS
        nodes.action = Marker.ADD
        nodes.scale.x = nodes.scale.y = 1.2
        nodes.color.r, nodes.color.g, nodes.color.b, nodes.color.a = 1.0, 0.75, 0.1, 0.9
        nodes.pose.orientation.w = 1.0

        n_pts = n_edges = 0
        for feat in data['features']:
            geom = feat['geometry']
            gtype = geom['type']
            coords = geom['coordinates']

            if gtype == 'Point':
                p = Point()
                p.x, p.y, p.z = float(coords[0]), float(coords[1]), 0.0
                nodes.points.append(p)
                n_pts += 1

            elif gtype in ('LineString', 'MultiLineString'):
                segs = [coords] if gtype == 'LineString' else coords
                for seg in segs:
                    # LINE_LIST consumes point pairs.
                    for a, b in zip(seg[:-1], seg[1:]):
                        pa, pb = Point(), Point()
                        pa.x, pa.y, pa.z = float(a[0]), float(a[1]), 0.0
                        pb.x, pb.y, pb.z = float(b[0]), float(b[1]), 0.0
                        edges.points.append(pa)
                        edges.points.append(pb)
                    n_edges += 1

        arr = MarkerArray()
        arr.markers = [edges, nodes]
        self.pub.publish(arr)
        self.get_logger().info(
            f'published route graph: {n_pts} nodes, {n_edges} edges '
            f'({len(edges.points)} line verts) in frame "{FRAME}" on /route_graph'
        )


def main():
    rclpy.init()
    node = RouteGraphViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
