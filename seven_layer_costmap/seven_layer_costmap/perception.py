"""Dependency-light perception primitives used by three-camera SVO processing."""

from dataclasses import dataclass
import math
from collections import deque
from typing import Dict, Optional

import numpy as np

try:
    import cv2
except ImportError:  # Unit-testable fallback for minimal environments.
    cv2 = None

from .core import GridSpec, TemporalVoxelGrid, inflate, rasterize_predictions


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class CameraSample:
    depth: np.ndarray
    bgr: np.ndarray
    intrinsic: np.ndarray
    stamp_s: float


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


def base_to_world(points, pose: Pose2D):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    result = pts.copy()
    result[:, 0] = pose.x + c * pts[:, 0] - s * pts[:, 1]
    result[:, 1] = pose.y + s * pts[:, 0] + c * pts[:, 1]
    return result


def world_to_base(points, pose: Pose2D):
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    dx, dy = pts[:, 0] - pose.x, pts[:, 1] - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    result = pts.copy()
    result[:, 0] = c * dx + s * dy
    result[:, 1] = -s * dx + c * dy
    return result


def _line_cells(start, end):
    """Integer Bresenham cells including start and end."""
    x0, y0 = start
    x1, y1 = end
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    error = dx + dy
    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def observed_mask_from_rays(spec: GridSpec, points, sensor_origins_base,
                            max_rays=1200, dilation_cells=1):
    """Rasterize camera-depth lines of sight into a 2-D observed-space mask.

    This mask is deliberately separate from occupancy: a valid stereo-depth ray
    means that space was observed, not that its endpoint is necessarily lethal.
    Cells between the three camera fields therefore remain unobserved instead of
    being mistaken for off-road or obstacle cells.
    """
    if max_rays <= 0 or dilation_cells < 0:
        raise ValueError('visibility ray limit must be positive and dilation nonnegative')
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    origins = np.asarray(sensor_origins_base, dtype=np.float32).reshape(-1, 3)
    observed = np.zeros(spec.shape, dtype=bool)
    if not len(pts):
        return observed
    if len(origins) == 1:
        origins = np.repeat(origins, len(pts), axis=0)
    if len(origins) != len(pts):
        raise ValueError('every visibility endpoint requires a sensor origin')

    height, width = spec.shape
    step = max(1, math.ceil(len(pts) / int(max_rays)))
    selected = np.arange(0, len(pts), step, dtype=np.int64)
    selected_points, selected_origins = pts[selected], origins[selected]
    starts = np.column_stack((
        np.floor((selected_origins[:, 0] + spec.width_m / 2) / spec.resolution),
        np.floor((selected_origins[:, 1] + spec.height_m / 2) / spec.resolution),
    )).astype(np.int32)
    ends = np.column_stack((
        np.floor((selected_points[:, 0] + spec.width_m / 2) / spec.resolution),
        np.floor((selected_points[:, 1] + spec.height_m / 2) / spec.resolution),
    )).astype(np.int32)
    if cv2 is not None:
        canvas = np.zeros(spec.shape, dtype=np.uint8)
        segments = np.stack((starts, ends), axis=1)
        cv2.polylines(canvas, segments, False, 1, 1, cv2.LINE_8)
        observed = canvas.astype(bool)
    else:
        for start, end in zip(starts, ends):
            for col, row in _line_cells(tuple(start), tuple(end)):
                if 0 <= row < height and 0 <= col < width:
                    observed[row, col] = True

    # Sparse ray sampling can leave one-cell pinholes at long range. A bounded
    # dilation closes only those sampling gaps; it does not bridge real camera
    # blind wedges, which are many cells wide.
    if cv2 is not None and dilation_cells:
        size = 2 * int(dilation_cells) + 1
        observed = cv2.dilate(observed.astype(np.uint8),
                              np.ones((size, size), np.uint8)).astype(bool)
    else:
        for _ in range(int(dilation_cells)):
            padded = np.pad(observed, 1, mode='constant')
            observed = np.logical_or.reduce([
                padded[row:row + height, col:col + width]
                for row in range(3) for col in range(3)
            ])
    return observed


def blind_spot_mask(spec: GridSpec, centers_deg=(-45.0, 45.0),
                    half_width_deg=18.0, min_range_m=1.5, max_range_m=12.0):
    """Return configurable front/side camera-gap wedges in ``base_link``."""
    if half_width_deg < 0 or min_range_m < 0 or max_range_m <= min_range_m:
        raise ValueError('invalid blind-spot angular or range limits')
    height, width = spec.shape
    xs = (np.arange(width) + 0.5) * spec.resolution - spec.width_m / 2
    ys = (np.arange(height) + 0.5) * spec.resolution - spec.height_m / 2
    xx, yy = np.meshgrid(xs, ys)
    radius = np.hypot(xx, yy)
    bearing = np.degrees(np.arctan2(yy, xx))
    mask = np.zeros(spec.shape, dtype=bool)
    for center in centers_deg:
        difference = (bearing - float(center) + 180.0) % 360.0 - 180.0
        mask |= np.abs(difference) <= float(half_width_deg)
    return mask & (radius >= float(min_range_m)) & (radius <= float(max_range_m))


class WorldOccupancyModel:
    """Sparse world-frame voxel history projected into a rolling vehicle grid."""

    def __init__(self, spec: GridSpec, persistence_s=2.0, voxel_m=0.20,
                 z_resolution=0.25, static_hits=8, max_voxels=750000,
                 max_clear_rays=75):
        if (persistence_s <= 0 or voxel_m <= 0 or z_resolution <= 0 or
                static_hits <= 0 or max_voxels <= 0 or max_clear_rays <= 0):
            raise ValueError('world occupancy limits and resolutions must be positive')
        self.spec = spec
        self.persistence_s = persistence_s
        self.voxel_m = voxel_m
        self.z_resolution = z_resolution
        self.static_hits = static_hits
        self.max_voxels = max_voxels
        self.max_clear_rays = max_clear_rays
        self.last_seen = {}
        self.hit_score = {}

    def _key(self, point):
        return (math.floor(point[0] / self.voxel_m),
                math.floor(point[1] / self.voxel_m),
                math.floor(point[2] / self.z_resolution))

    def observe(self, base_points, pose: Pose2D, sensor_origins_base, stamp_s):
        points = np.asarray(base_points, dtype=np.float32).reshape(-1, 3)
        if not len(points):
            self.prune(stamp_s)
            return
        world = base_to_world(points, pose)
        origins_world = base_to_world(np.asarray(sensor_origins_base).reshape(-1, 3), pose)
        # Clear only a sampled subset of rays to bound Python cost.
        sample_step = max(1, math.ceil(len(world) / self.max_clear_rays))
        for index in range(0, len(world), sample_step):
            endpoint = world[index]
            origin = origins_world[index % len(origins_world)]
            start_xy = (math.floor(origin[0] / self.voxel_m),
                        math.floor(origin[1] / self.voxel_m))
            end_xy = (math.floor(endpoint[0] / self.voxel_m),
                      math.floor(endpoint[1] / self.voxel_m))
            clear_xy = set(_line_cells(start_xy, end_xy)[:-1])
            if clear_xy:
                for cell_x, cell_y in clear_xy:
                    for cell_z in range(-16, 17):
                        key = (cell_x, cell_y, cell_z)
                        self.last_seen.pop(key, None)
                        self.hit_score.pop(key, None)
        observed = set()
        for point in world:
            key = self._key(point)
            self.last_seen[key] = stamp_s
            observed.add(key)
        for key in list(self.hit_score):
            if key not in observed:
                self.hit_score[key] = max(0, self.hit_score[key] - 1)
                if self.hit_score[key] == 0:
                    self.hit_score.pop(key, None)
        for key in observed:
            self.hit_score[key] = min(self.static_hits * 2, self.hit_score.get(key, 0) + 2)
        self.prune(stamp_s)

    def prune(self, stamp_s):
        expired = [key for key, seen in self.last_seen.items()
                   if stamp_s - seen > self.persistence_s]
        for key in expired:
            self.last_seen.pop(key, None)
        if len(self.last_seen) > self.max_voxels:
            oldest = sorted(self.last_seen, key=self.last_seen.get)[:len(self.last_seen)-self.max_voxels]
            for key in oldest:
                self.last_seen.pop(key, None)
                self.hit_score.pop(key, None)

    def project(self, pose: Pose2D, stamp_s):
        self.prune(stamp_s)
        transient = np.zeros(self.spec.shape, dtype=np.uint8)
        static = np.zeros(self.spec.shape, dtype=np.uint8)
        if not self.last_seen:
            return static, transient
        keys = list(self.last_seen)
        world = np.array([((x + 0.5) * self.voxel_m,
                           (y + 0.5) * self.voxel_m,
                           (z + 0.5) * self.z_resolution) for x, y, z in keys])
        base = world_to_base(world, pose)
        cols = ((base[:, 0] + self.spec.width_m / 2) / self.spec.resolution).astype(int)
        rows = ((base[:, 1] + self.spec.height_m / 2) / self.spec.resolution).astype(int)
        valid = ((cols >= 0) & (cols < self.spec.shape[1]) &
                 (rows >= 0) & (rows < self.spec.shape[0]))
        scores = np.fromiter((self.hit_score.get(key, 0) for key in keys),
                             dtype=np.int16, count=len(keys))
        static_mask = valid & (scores >= self.static_hits)
        transient_mask = valid & ~static_mask
        static[rows[static_mask], cols[static_mask]] = 100
        transient[rows[transient_mask], cols[transient_mask]] = 100
        return static, transient


class ThreeCameraSynchronizer:
    """Bounded ordered synchronizer for independently decoded SVO streams."""

    def __init__(self, names=('front', 'left', 'right'), max_skew_s=0.050,
                 queue_size=90):
        if max_skew_s < 0 or queue_size <= 0:
            raise ValueError('sync skew must be nonnegative and queue size positive')
        self.names = tuple(names)
        self.max_skew_s = max_skew_s
        self.queue_size = int(queue_size)
        self.samples: Dict[str, CameraSample] = {}
        self.queues = {name: deque(maxlen=self.queue_size) for name in self.names}
        self.last_emitted_s = -math.inf
        self.dropped = 0

    def update(self, name: str, sample: CameraSample) -> None:
        if name not in self.names:
            raise KeyError(name)
        queue = self.queues[name]
        if queue and sample.stamp_s <= queue[-1].stamp_s:
            return
        if len(queue) == queue.maxlen:
            self.dropped += 1
        queue.append(sample)
        self.samples[name] = sample

    def take(self) -> Optional[Dict[str, CameraSample]]:
        while all(self.queues[name] for name in self.names):
            heads = {name: self.queues[name][0] for name in self.names}
            stamps = {name: sample.stamp_s for name, sample in heads.items()}
            minimum, maximum = min(stamps.values()), max(stamps.values())
            if maximum - minimum <= self.max_skew_s and minimum > self.last_emitted_s:
                self.last_emitted_s = minimum
                return {name: self.queues[name].popleft() for name in self.names}
            # A head older than the newest head by more than the tolerance can
            # never match any later sample from the other ordered streams.
            oldest = [name for name, stamp in stamps.items() if stamp == minimum]
            for name in oldest:
                self.queues[name].popleft()
                self.dropped += 1
        return None


class SkewMonitor:
    """Bounded synchronization statistics for diagnostics and regression tests."""

    def __init__(self, limit_s, window=100):
        self.limit_s = limit_s
        self.values = deque(maxlen=window)
        self.violations = 0

    def observe(self, stamps):
        values = list(stamps)
        skew = max(values) - min(values)
        self.values.append(skew)
        if skew > self.limit_s:
            self.violations += 1
        return skew

    def summary(self):
        if not self.values:
            return {'current_s': 0.0, 'mean_s': 0.0, 'max_s': 0.0, 'violations': 0}
        return {'current_s': self.values[-1], 'mean_s': sum(self.values) / len(self.values),
                'max_s': max(self.values), 'violations': self.violations}


def inflation_radius_for_speed(base_radius_m, speed_mps, reaction_time_s,
                               max_extra_m):
    """Conservative bounded longitudinal-speed proxy for radial inflation."""
    return float(base_radius_m) + min(float(max_extra_m),
                                      max(0.0, float(speed_mps)) * float(reaction_time_s))


def rotation_matrix_from_rpy(rpy):
    """Return the REP-103 fixed-axis roll/pitch/yaw rotation matrix."""
    values = np.asarray(rpy, dtype=np.float64).reshape(-1)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError('rpy must contain three finite values')
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def depth_to_base_points(depth, intrinsic, translation, yaw=0.0, stride=4,
                         min_depth=0.5, max_depth=25.0, rpy=None):
    """Back-project ZED depth and apply the calibrated 6-DoF camera mount.

    ``yaw`` remains for compatibility with the initial milestone. New callers
    should provide ``rpy=[roll, pitch, yaw]`` so camera tilt is not discarded.
    """
    z = np.asarray(depth, dtype=np.float32)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if z.ndim != 2:
        raise ValueError('depth image must be two-dimensional')
    if stride <= 0 or min_depth < 0 or max_depth <= min_depth:
        raise ValueError('invalid depth sampling or range parameters')
    if len(translation) != 3 or not np.isfinite(translation).all():
        raise ValueError('translation must contain three finite values')
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('camera intrinsics must contain positive focal lengths')
    rows = np.arange(0, z.shape[0], stride)
    cols = np.arange(0, z.shape[1], stride)
    uu, vv = np.meshgrid(cols, rows)
    forward = z[vv, uu]
    valid = np.isfinite(forward) & (forward >= min_depth) & (forward <= max_depth)
    forward = forward[valid]
    optical_right = (uu[valid] - k[0, 2]) / k[0, 0] * forward
    optical_down = (vv[valid] - k[1, 2]) / k[1, 1] * forward
    # Optical (right, down, forward) -> camera REP-103 (forward, left, up).
    x, y, z_up = forward, -optical_right, -optical_down
    camera_points = np.column_stack((x, y, z_up))
    rotation = rotation_matrix_from_rpy((0.0, 0.0, yaw) if rpy is None else rpy)
    return (camera_points @ rotation.T + translation).astype(np.float32)


def remove_ground(points, band_m=0.18):
    """Remove the dominant near-horizontal height band from a calibrated point set."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    candidates = pts[(pts[:, 2] > -3.0) & (pts[:, 2] < 1.0)]
    if len(candidates) < 50:
        return pts
    counts, edges = np.histogram(candidates[:, 2], bins=np.arange(-3.0, 1.05, 0.05))
    ground_z = (edges[int(np.argmax(counts))] + edges[int(np.argmax(counts)) + 1]) / 2
    return pts[np.abs(pts[:, 2] - ground_z) > band_m]


def obstacle_grid(spec: GridSpec, points, z_min=-1.5, z_max=2.8,
                  min_points_per_cell=1):
    grid = np.zeros(spec.shape, dtype=np.uint8)
    if min_points_per_cell <= 0:
        raise ValueError('min_points_per_cell must be positive')
    pts = np.asarray(points).reshape(-1, 3)
    pts = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]
    if not len(pts):
        return grid
    cols = ((pts[:, 0] + spec.width_m / 2) / spec.resolution).astype(int)
    rows = ((pts[:, 1] + spec.height_m / 2) / spec.resolution).astype(int)
    valid = ((cols >= 0) & (cols < spec.shape[1]) &
             (rows >= 0) & (rows < spec.shape[0]))
    counts = np.zeros(spec.shape, dtype=np.uint16)
    np.add.at(counts, (rows[valid], cols[valid]), 1)
    grid[counts >= int(min_points_per_cell)] = 100
    return grid


def vision_bev_grid(spec: GridSpec, points, sensor_origins_base,
                    visibility_max_rays=2400, visibility_dilation_cells=1,
                    ground_band_m=0.18, obstacle_z_min=-1.5,
                    obstacle_z_max=2.8, min_points_per_cell=2):
    """Build an instantaneous camera-only BEV with unknown/free/occupied cells.

    Unknown is ``-1``, observed free space is ``0``, and obstacle cells are
    ``100``. No odometry or cross-frame accumulation is used, preventing stale
    geometry from being smeared when recordings were captured from a moving rig.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    observed = observed_mask_from_rays(
        spec, pts, sensor_origins_base, max_rays=visibility_max_rays,
        dilation_cells=visibility_dilation_cells)
    obstacles = remove_ground(pts, band_m=ground_band_m)
    obstacles = obstacles[(obstacles[:, 2] >= obstacle_z_min) &
                          (obstacles[:, 2] <= obstacle_z_max)]
    occupied = obstacle_grid(
        spec, obstacles, z_min=obstacle_z_min, z_max=obstacle_z_max,
        min_points_per_cell=min_points_per_cell)
    grid = np.full(spec.shape, -1, dtype=np.int8)
    grid[observed] = 0
    grid[occupied >= 100] = 100
    return grid, obstacles


def colorize_bev(grid):
    """Colorize a BEV for an image panel with vehicle-forward pointing up."""
    values = np.asarray(grid)
    image = np.zeros((*values.shape, 3), dtype=np.uint8)
    image[values < 0] = (55, 55, 55)       # unknown: gray
    image[values == 0] = (35, 95, 35)      # observed free: green
    image[values >= 100] = (20, 20, 235)   # occupied: red (BGR)
    # Grid columns are +X and rows are +Y. Rotate/flip for conventional BEV:
    # +X forward is up and +Y left is left.
    return np.ascontiguousarray(np.rot90(image, 1)[:, ::-1])


class PersistenceSeparator:
    """Separates repeatedly occupied cells from transient observations."""

    def __init__(self, shape, static_hits=8, decay=1):
        self.score = np.zeros(shape, dtype=np.int16)
        self.static_hits = static_hits
        self.decay = decay

    def update(self, occupied):
        mask = np.asarray(occupied) >= 99
        self.score = np.maximum(0, self.score - self.decay)
        self.score[mask] = np.minimum(self.static_hits * 2, self.score[mask] + 2)
        static = np.where(self.score >= self.static_hits, 100, 0).astype(np.uint8)
        transient = np.where(mask & (self.score < self.static_hits), 100, 0).astype(np.uint8)
        return static, transient


def connected_centroids(grid, spec: GridSpec, min_cells=2):
    """Return grid-component centroids without scipy/OpenCV dependencies."""
    active = np.asarray(grid) >= 99
    seen = np.zeros_like(active, dtype=bool)
    centroids = []
    h, w = active.shape
    for row, col in np.argwhere(active):
        if seen[row, col]:
            continue
        stack, cells = [(row, col)], []
        seen[row, col] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and active[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        if len(cells) >= min_cells:
            mean_r, mean_c = np.mean(cells, axis=0)
            centroids.append(((mean_c + 0.5) * spec.resolution - spec.width_m / 2,
                              (mean_r + 0.5) * spec.resolution - spec.height_m / 2))
    return centroids


class CentroidTracker:
    def __init__(self, gate_m=3.0):
        self.gate_m = gate_m
        self.previous = []
        self.previous_stamp = None

    def update(self, centroids, stamp_s):
        tracks = []
        dt = None if self.previous_stamp is None else stamp_s - self.previous_stamp
        for x, y in centroids:
            velocity = (0.0, 0.0)
            if dt and dt > 1e-3 and self.previous:
                px, py = min(self.previous, key=lambda p: math.hypot(x - p[0], y - p[1]))
                if math.hypot(x - px, y - py) <= self.gate_m:
                    velocity = ((x - px) / dt, (y - py) / dt)
            tracks.append((x, y, velocity[0], velocity[1]))
        self.previous, self.previous_stamp = list(centroids), stamp_s
        return tracks


def vision_lane_layer(spec: GridSpec, bgr, half_width_m=2.0, max_range_m=25.0,
                      observed_mask=None, blind_centers_deg=(-45.0, 45.0),
                      blind_half_width_deg=18.0, blind_min_range_m=1.5,
                      blind_max_range_m=12.0, blind_unknown_cost=25,
                      blind_clear_cost=0, off_corridor_cost=95):
    """Conservative local drivable corridor; image markings adjust its lateral center."""
    image = np.asarray(bgr, dtype=np.float32)
    lower = image[image.shape[0] // 2:]
    intensity = lower.mean(axis=2)
    spread = lower.max(axis=2) - lower.min(axis=2)
    marking = (intensity > 175) & (spread < 55)
    columns = np.argwhere(marking)[:, 1] if marking.any() else np.array([])
    offset = 0.0
    if len(columns) >= 20:
        normalized = float(np.median(columns) / image.shape[1] - 0.5)
        offset = float(np.clip(-normalized * 3.0, -1.5, 1.5))
    h, w = spec.shape
    for name, value in (('blind_unknown_cost', blind_unknown_cost),
                        ('blind_clear_cost', blind_clear_cost),
                        ('off_corridor_cost', off_corridor_cost)):
        if not 0 <= int(value) <= 100:
            raise ValueError(f'{name} must be in [0, 100]')
    costs = np.full((h, w), int(off_corridor_cost), dtype=np.uint8)
    xs = (np.arange(w) + 0.5) * spec.resolution - spec.width_m / 2
    ys = (np.arange(h) + 0.5) * spec.resolution - spec.height_m / 2
    corridor = ((xs[None, :] >= 0) & (xs[None, :] <= max_range_m) &
                (np.abs(ys[:, None] - offset) <= half_width_m))
    costs[corridor] = 0
    if observed_mask is not None:
        observed = np.asarray(observed_mask, dtype=bool)
        if observed.shape != spec.shape:
            raise ValueError('observed_mask geometry does not match costmap')
        blind = blind_spot_mask(
            spec, blind_centers_deg, blind_half_width_deg,
            blind_min_range_m, blind_max_range_m)
        costs[blind & ~observed] = int(blind_unknown_cost)
        costs[blind & observed] = int(blind_clear_cost)
    return costs


def traffic_regulation_layer(spec: GridSpec, images, stop_distance_m=8.0):
    """Conservative red-light visual gate; trained signal/sign detector can replace it."""
    red = False
    for bgr in images:
        image = np.asarray(bgr, dtype=np.float32)
        upper = image[:max(1, image.shape[0] // 2)]
        red_pixels = (upper[:, :, 2] > 150) & (upper[:, :, 2] > 1.4 * upper[:, :, 1]) & (
            upper[:, :, 2] > 1.4 * upper[:, :, 0])
        red = red or float(red_pixels.mean()) > 0.0005
    grid = np.zeros(spec.shape, dtype=np.uint8)
    if red:
        col = int((stop_distance_m + spec.width_m / 2) / spec.resolution)
        center_row = spec.shape[0] // 2
        half = max(1, int(3.0 / spec.resolution))
        grid[center_row - half:center_row + half + 1, max(0, col - 1):col + 2] = 100
    return grid


def derive_layers(spec, points, front_bgr, all_images, occupancy_model, tracker, pose, stamp_s,
                  sensor_origins_base,
                  inflation_radius_m=2.5, visibility_max_rays=1200,
                  visibility_dilation_cells=1, blind_centers_deg=(-45.0, 45.0),
                  blind_half_width_deg=18.0, blind_min_range_m=1.5,
                  blind_max_range_m=12.0, blind_unknown_cost=25,
                  blind_clear_cost=0, temporal_memory=True,
                  enable_prediction=True):
    observed = observed_mask_from_rays(
        spec, points, sensor_origins_base, visibility_max_rays,
        visibility_dilation_cells)
    obstacle_points = remove_ground(points)
    obstacle_points = obstacle_points[(obstacle_points[:, 2] >= -1.5) &
                                      (obstacle_points[:, 2] <= 2.8)]
    current = obstacle_grid(spec, obstacle_points)
    if temporal_memory:
        occupancy_model.observe(obstacle_points, pose, sensor_origins_base, stamp_s)
        static, temporal = occupancy_model.project(pose, stamp_s)
    else:
        static = np.zeros(spec.shape, dtype=np.uint8)
        temporal = current
    if enable_prediction:
        centroids = connected_centroids(current, spec)
        tracks = tracker.update(centroids, stamp_s)
        prediction = rasterize_predictions(spec, tracks)
    else:
        prediction = np.zeros(spec.shape, dtype=np.uint8)
    combined = np.maximum.reduce((static, temporal, prediction))
    return {
        'lanelet': vision_lane_layer(
            spec, front_bgr, observed_mask=observed,
            blind_centers_deg=blind_centers_deg,
            blind_half_width_deg=blind_half_width_deg,
            blind_min_range_m=blind_min_range_m,
            blind_max_range_m=blind_max_range_m,
            blind_unknown_cost=blind_unknown_cost,
            blind_clear_cost=blind_clear_cost),
        'static_obstacle': static,
        'spatio_temporal_voxel': temporal,
        'prediction': prediction,
        'inflation': inflate(combined, inflation_radius_m, spec.resolution, 1.2),
        'traffic_regulation': traffic_regulation_layer(spec, all_images),
    }
