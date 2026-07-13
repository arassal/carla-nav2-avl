"""Dependency-light perception primitives used by three-camera SVO processing."""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import numpy as np

from .core import GridSpec, TemporalVoxelGrid, inflate, rasterize_predictions


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


@dataclass
class CameraSample:
    depth: np.ndarray
    bgr: np.ndarray
    intrinsic: np.ndarray
    stamp_s: float


class ThreeCameraSynchronizer:
    """Latest-sample synchronizer that never reuses a set or accepts excess skew."""

    def __init__(self, names=('front', 'left', 'right'), max_skew_s=0.050):
        self.names = tuple(names)
        self.max_skew_s = max_skew_s
        self.samples: Dict[str, CameraSample] = {}
        self.last_emitted_s = -math.inf

    def update(self, name: str, sample: CameraSample) -> None:
        if name not in self.names:
            raise KeyError(name)
        self.samples[name] = sample

    def take(self) -> Optional[Dict[str, CameraSample]]:
        if any(name not in self.samples for name in self.names):
            return None
        stamps = [self.samples[name].stamp_s for name in self.names]
        if max(stamps) - min(stamps) > self.max_skew_s or min(stamps) <= self.last_emitted_s:
            return None
        self.last_emitted_s = min(stamps)
        return {name: self.samples[name] for name in self.names}


def depth_to_base_points(depth, intrinsic, translation, yaw=0.0, stride=4,
                         min_depth=0.5, max_depth=25.0):
    """Back-project ZED depth and transform optical XYZ to REP-103 base coordinates."""
    z = np.asarray(depth, dtype=np.float32)
    k = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
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
    c, s = math.cos(yaw), math.sin(yaw)
    base_x = c * x - s * y + translation[0]
    base_y = s * x + c * y + translation[1]
    base_z = z_up + translation[2]
    return np.column_stack((base_x, base_y, base_z)).astype(np.float32)


def remove_ground(points, band_m=0.18):
    """Remove the dominant near-horizontal height band from a calibrated point set."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    candidates = pts[(pts[:, 2] > -3.0) & (pts[:, 2] < 1.0)]
    if len(candidates) < 50:
        return pts
    counts, edges = np.histogram(candidates[:, 2], bins=np.arange(-3.0, 1.05, 0.05))
    ground_z = (edges[int(np.argmax(counts))] + edges[int(np.argmax(counts)) + 1]) / 2
    return pts[np.abs(pts[:, 2] - ground_z) > band_m]


def obstacle_grid(spec: GridSpec, points, z_min=-1.5, z_max=2.8):
    grid = np.zeros(spec.shape, dtype=np.uint8)
    pts = np.asarray(points).reshape(-1, 3)
    pts = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)]
    if not len(pts):
        return grid
    cols = ((pts[:, 0] + spec.width_m / 2) / spec.resolution).astype(int)
    rows = ((pts[:, 1] + spec.height_m / 2) / spec.resolution).astype(int)
    valid = ((cols >= 0) & (cols < spec.shape[1]) &
             (rows >= 0) & (rows < spec.shape[0]))
    grid[rows[valid], cols[valid]] = 100
    return grid


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


def vision_lane_layer(spec: GridSpec, bgr, half_width_m=2.0, max_range_m=25.0):
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
    costs = np.full((h, w), 95, dtype=np.uint8)
    xs = (np.arange(w) + 0.5) * spec.resolution - spec.width_m / 2
    ys = (np.arange(h) + 0.5) * spec.resolution - spec.height_m / 2
    corridor = ((xs[None, :] >= 0) & (xs[None, :] <= max_range_m) &
                (np.abs(ys[:, None] - offset) <= half_width_m))
    costs[corridor] = 0
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


def derive_layers(spec, points, front_bgr, all_images, separator, tracker, voxel_grid, stamp_s,
                  inflation_radius_m=2.5):
    obstacle_points = remove_ground(points)
    occupied = obstacle_grid(spec, obstacle_points)
    static, transient = separator.update(occupied)
    obstacle_points = obstacle_points[(obstacle_points[:, 2] >= -1.5) &
                                      (obstacle_points[:, 2] <= 2.8)]
    voxel_grid.observe(obstacle_points, stamp_s)
    temporal = voxel_grid.project(stamp_s)
    centroids = connected_centroids(transient, spec)
    tracks = tracker.update(centroids, stamp_s)
    prediction = rasterize_predictions(spec, tracks)
    combined = np.maximum.reduce((static, temporal, prediction))
    return {
        'lanelet': vision_lane_layer(spec, front_bgr),
        'static_obstacle': static,
        'spatio_temporal_voxel': temporal,
        'prediction': prediction,
        'inflation': inflate(combined, inflation_radius_m, spec.resolution, 1.2),
        'traffic_regulation': traffic_regulation_layer(spec, all_images),
    }
