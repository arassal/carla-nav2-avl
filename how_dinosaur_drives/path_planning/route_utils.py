"""Geometry helpers for campus route following."""

import math
from typing import Iterable, List, Optional, Sequence, Tuple


Point2D = Tuple[float, float]


def polyline_length(points: Sequence[Point2D]) -> float:
    """Return the cumulative Euclidean length of a polyline."""
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points[:-1], points[1:]))


def resample_polyline(
        points: Iterable[Point2D], spacing_m: float) -> List[Point2D]:
    """Return spaced route points while retaining the endpoint."""
    source = list(points)
    if not source:
        return []
    if spacing_m <= 0.0:
        raise ValueError('spacing_m must be positive')

    sampled = [source[0]]
    distance_to_next = spacing_m
    segment_start = source[0]

    for segment_end in source[1:]:
        dx = segment_end[0] - segment_start[0]
        dy = segment_end[1] - segment_start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-9:
            segment_start = segment_end
            continue

        while segment_length >= distance_to_next:
            ratio = distance_to_next / segment_length
            point = (
                segment_start[0] + ratio * dx,
                segment_start[1] + ratio * dy,
            )
            sampled.append(point)
            segment_start = point
            dx = segment_end[0] - segment_start[0]
            dy = segment_end[1] - segment_start[1]
            segment_length = math.hypot(dx, dy)
            distance_to_next = spacing_m

        distance_to_next -= segment_length
        segment_start = segment_end

    if math.hypot(sampled[-1][0] - source[-1][0],
                  sampled[-1][1] - source[-1][1]) > 1e-6:
        sampled.append(source[-1])
    return sampled


def turn_angle_degrees(points: Sequence[Point2D], index: int) -> float:
    """Return the unsigned heading change at an interior route point."""
    if index <= 0 or index >= len(points) - 1:
        return 0.0
    before = math.atan2(points[index][1] - points[index - 1][1],
                        points[index][0] - points[index - 1][0])
    after = math.atan2(points[index + 1][1] - points[index][1],
                       points[index + 1][0] - points[index][0])
    delta = math.atan2(math.sin(after - before), math.cos(after - before))
    return abs(math.degrees(delta))


def handoff_radius(points: Sequence[Point2D], index: int,
                   straight_radius_m: float, turn_radius_m: float,
                   sharp_turn_deg: float) -> float:
    """Use a tighter handoff near turns so Nav2 does not cut road corners."""
    if turn_angle_degrees(points, index) >= sharp_turn_deg:
        return turn_radius_m
    return straight_radius_m


def yaw_toward(points: Sequence[Point2D], index: int) -> float:
    """Return the route tangent heading at a sampled point."""
    if len(points) < 2:
        return 0.0
    if index < len(points) - 1:
        a, b = points[index], points[index + 1]
    else:
        a, b = points[index - 1], points[index]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return planar yaw from a quaternion."""
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def map_to_robot(
        point: Point2D,
        robot_pose: Tuple[float, float, float]) -> Point2D:
    """Transform one map-frame point into the planar robot frame."""
    px, py, yaw = robot_pose
    dx, dy = point[0] - px, point[1] - py
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def max_grid_cost_near(
        data: Sequence[int], width: int, height: int, resolution: float,
        origin: Point2D, point: Point2D, radius_m: float) -> Optional[int]:
    """Return the maximum grid cost in a metric disk, or None if off-grid."""
    if width <= 0 or height <= 0 or resolution <= 0.0 or radius_m < 0.0:
        raise ValueError('invalid grid geometry')
    if len(data) != width * height:
        raise ValueError('grid data length does not match dimensions')

    center_col = math.floor((point[0] - origin[0]) / resolution)
    center_row = math.floor((point[1] - origin[1]) / resolution)
    if not (0 <= center_col < width and 0 <= center_row < height):
        return None

    cell_radius = math.ceil(radius_m / resolution)
    limit_sq = (radius_m + 0.5 * resolution) ** 2
    costs = []
    for row in range(max(0, center_row - cell_radius),
                     min(height, center_row + cell_radius + 1)):
        cell_y = origin[1] + (row + 0.5) * resolution
        for col in range(max(0, center_col - cell_radius),
                         min(width, center_col + cell_radius + 1)):
            cell_x = origin[0] + (col + 0.5) * resolution
            if ((cell_x - point[0]) ** 2 + (cell_y - point[1]) ** 2
                    <= limit_sq):
                costs.append(int(data[row * width + col]))
    return max(costs) if costs else int(data[center_row * width + center_col])
