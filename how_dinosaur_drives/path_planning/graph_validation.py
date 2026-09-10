"""Offline integrity and route-probe checks for nav2_route GeoJSON graphs."""

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Point2D = Tuple[float, float]
WeightedEdge = Tuple[int, float]


@dataclass
class GraphReport:
    """Structured campus graph validation result."""

    nodes: int = 0
    edges: int = 0
    weak_components: int = 0
    largest_weak_component: int = 0
    strong_components: int = 0
    largest_strong_component: int = 0
    route_probes: int = 0
    route_probe_distance_m: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether every required integrity check passed."""
        return not self.errors


def validate_graph(document: dict, route_probe_count: int = 20) -> GraphReport:
    """Validate graph schema, geometry, connectivity, and sampled routes."""
    report = GraphReport()
    features = document.get('features')
    if not isinstance(features, list):
        report.errors.append('document.features must be a list')
        return report

    nodes: Dict[int, Point2D] = {}
    node_coordinates = set()
    raw_edges = []
    feature_ids = set()

    for index, feature in enumerate(features):
        properties = feature.get('properties', {})
        geometry = feature.get('geometry', {})
        feature_id = properties.get('id')
        if not isinstance(feature_id, int):
            report.errors.append(f'feature {index} has no integer id')
            continue
        if feature_id in feature_ids:
            report.errors.append(f'duplicate feature id {feature_id}')
        feature_ids.add(feature_id)

        geometry_type = geometry.get('type')
        if geometry_type == 'Point':
            coordinates = geometry.get('coordinates', [])
            if len(coordinates) < 2:
                report.errors.append(f'node {feature_id} has invalid coordinates')
                continue
            point = (float(coordinates[0]), float(coordinates[1]))
            if not all(math.isfinite(value) for value in point):
                report.errors.append(f'node {feature_id} has non-finite coordinates')
                continue
            if point in node_coordinates:
                report.errors.append(f'duplicate node coordinate {point}')
            node_coordinates.add(point)
            nodes[feature_id] = point
            if properties.get('frame') != 'map':
                report.errors.append(f'node {feature_id} frame is not map')
        elif geometry_type == 'MultiLineString':
            raw_edges.append((feature_id, properties, geometry))
        else:
            report.errors.append(
                f'feature {feature_id} has unsupported geometry {geometry_type}')

    report.nodes = len(nodes)
    report.edges = len(raw_edges)
    adjacency = {node_id: [] for node_id in nodes}
    reverse = {node_id: [] for node_id in nodes}
    undirected = {node_id: [] for node_id in nodes}
    seen_edges = set()

    for edge_id, properties, geometry in raw_edges:
        start = properties.get('startid')
        end = properties.get('endid')
        if start not in nodes or end not in nodes:
            report.errors.append(
                f'edge {edge_id} references missing endpoint {start}->{end}')
            continue
        if start == end:
            report.errors.append(f'edge {edge_id} is a self-loop at {start}')
            continue
        if (start, end) in seen_edges:
            report.errors.append(f'duplicate directed edge {start}->{end}')
            continue
        seen_edges.add((start, end))

        coordinates = geometry.get('coordinates', [])
        if (not coordinates or len(coordinates[0]) < 2 or
                len(coordinates[0][0]) < 2 or len(coordinates[0][-1]) < 2):
            report.errors.append(f'edge {edge_id} has invalid geometry')
            continue
        geometry_start = tuple(float(v) for v in coordinates[0][0][:2])
        geometry_end = tuple(float(v) for v in coordinates[0][-1][:2])
        if _distance(geometry_start, nodes[start]) > 0.01:
            report.errors.append(f'edge {edge_id} start geometry mismatch')
        if _distance(geometry_end, nodes[end]) > 0.01:
            report.errors.append(f'edge {edge_id} end geometry mismatch')

        cost = float(properties.get('cost', _distance(nodes[start], nodes[end])))
        if not math.isfinite(cost) or cost <= 0.0:
            report.errors.append(f'edge {edge_id} has invalid cost {cost}')
            continue
        adjacency[start].append((end, cost))
        reverse[end].append((start, cost))
        undirected[start].append(end)
        undirected[end].append(start)

    if not nodes:
        report.errors.append('graph has no nodes')
        return report
    if not raw_edges:
        report.errors.append('graph has no edges')
        return report

    weak = _components(undirected)
    strong = _strong_components(adjacency, reverse)
    report.weak_components = len(weak)
    report.largest_weak_component = max(map(len, weak), default=0)
    report.strong_components = len(strong)
    report.largest_strong_component = max(map(len, strong), default=0)
    if len(weak) > 1:
        report.warnings.append(
            f'graph has {len(weak)} disconnected road components')
    if report.largest_strong_component < report.nodes:
        fringe = report.nodes - report.largest_strong_component
        report.warnings.append(
            f'{fringe} nodes are outside the bidirectionally routable core')

    largest = max(strong, key=len, default=[])
    probes = _probe_pairs(largest, route_probe_count)
    for start, end in probes:
        distance = _shortest_distance(adjacency, start, end)
        if not math.isfinite(distance):
            report.errors.append(f'route probe failed: {start}->{end}')
        else:
            report.route_probes += 1
            report.route_probe_distance_m += distance
    return report


def main(args=None) -> int:
    """Validate a graph file and return a shell-friendly status code."""
    parser = argparse.ArgumentParser()
    parser.add_argument('graph', type=Path)
    parser.add_argument('--route-probes', type=int, default=20)
    parsed = parser.parse_args(args)

    with parsed.graph.open() as source:
        report = validate_graph(json.load(source), parsed.route_probes)
    print(f'nodes={report.nodes} edges={report.edges}')
    print(
        f'weak_components={report.weak_components} '
        f'largest_weak={report.largest_weak_component}')
    print(
        f'strong_components={report.strong_components} '
        f'largest_strong={report.largest_strong_component}')
    print(
        f'route_probes={report.route_probes} '
        f'total_probe_distance_m={report.route_probe_distance_m:.1f}')
    for warning in report.warnings:
        print(f'WARNING: {warning}')
    for error in report.errors:
        print(f'ERROR: {error}')
    print('PASS' if report.ok else 'FAIL')
    return 0 if report.ok else 1


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _components(adjacency: Dict[int, Iterable[int]]) -> List[List[int]]:
    remaining = set(adjacency)
    result = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.remove(seed)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _strong_components(
        adjacency: Dict[int, Sequence[WeightedEdge]],
        reverse: Dict[int, Sequence[WeightedEdge]]) -> List[List[int]]:
    visited = set()
    order = []
    for seed in sorted(adjacency):
        if seed in visited:
            continue
        visited.add(seed)
        stack = [(seed, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            stack.append((node, True))
            for neighbor, _cost in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, False))

    remaining = set(adjacency)
    result = []
    for seed in reversed(order):
        if seed not in remaining:
            continue
        remaining.remove(seed)
        stack = [seed]
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _cost in reverse[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _probe_pairs(nodes: Sequence[int], count: int) -> List[Tuple[int, int]]:
    ordered = sorted(nodes)
    if len(ordered) < 2 or count <= 0:
        return []
    pairs = []
    stride = max(1, len(ordered) // count)
    offset = max(1, len(ordered) // 2)
    for index in range(0, len(ordered), stride):
        start = ordered[index]
        end = ordered[(index + offset) % len(ordered)]
        if start != end:
            pairs.append((start, end))
        if len(pairs) >= count:
            break
    return pairs


def _shortest_distance(
        adjacency: Dict[int, Sequence[WeightedEdge]],
        start: int, goal: int) -> float:
    best = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, node = heapq.heappop(queue)
        if node == goal:
            return distance
        if distance != best[node]:
            continue
        for neighbor, cost in adjacency[node]:
            candidate = distance + cost
            if candidate < best.get(neighbor, math.inf):
                best[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return math.inf


if __name__ == '__main__':
    raise SystemExit(main())
