"""ROS-independent costmap algorithms, intentionally unit-testable without ROS."""

from dataclasses import dataclass
import heapq
import math
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # CI/reference environments may intentionally omit OpenCV.
    cv2 = None

LAYER_NAMES = (
    'lanelet', 'static_obstacle', 'spatio_temporal_voxel', 'prediction',
    'inflation',
)


@dataclass(frozen=True)
class GridSpec:
    width_m: float = 60.0
    height_m: float = 60.0
    resolution: float = 0.20

    def __post_init__(self):
        if self.width_m <= 0 or self.height_m <= 0 or self.resolution <= 0:
            raise ValueError('grid dimensions and resolution must be positive')
        for dimension in (self.width_m, self.height_m):
            cells = dimension / self.resolution
            if not math.isclose(cells, round(cells), abs_tol=1e-9):
                raise ValueError('grid dimensions must be exact multiples of resolution')

    @property
    def shape(self) -> Tuple[int, int]:
        return (round(self.height_m / self.resolution),
                round(self.width_m / self.resolution))


def normalize_layer(data: Iterable[int], shape: Tuple[int, int]) -> np.ndarray:
    """Convert ROS occupancy values to uint8 costs; unknown (-1) becomes zero."""
    grid = np.asarray(list(data), dtype=np.int16).reshape(shape)
    return np.where(grid < 0, 0, np.clip(grid, 0, 100)).astype(np.uint8)


def fuse_layers(layers: Dict[str, np.ndarray], weights=None) -> np.ndarray:
    """Fuse seven independently observable layers using weighted max cost."""
    if not layers:
        raise ValueError('at least one layer is required')
    weights = weights or {}
    first = next(iter(layers.values()))
    result = np.zeros_like(first, dtype=np.uint8)
    for name in LAYER_NAMES:
        if name not in layers:
            continue
        layer = np.asarray(layers[name])
        if layer.shape != result.shape:
            raise ValueError(f'{name} shape {layer.shape} != {result.shape}')
        weighted = np.clip(layer.astype(np.float32) * weights.get(name, 1.0), 0, 100)
        result = np.maximum(result, weighted.astype(np.uint8))
    return result


def inflate(lethal_grid: np.ndarray, radius_m: float, resolution: float,
            decay: float = 3.0) -> np.ndarray:
    """Multi-source bounded inflation with exponential decay in O(cells log cells)."""
    if radius_m < 0 or resolution <= 0 or decay < 0:
        raise ValueError('radius and decay must be nonnegative; resolution must be positive')
    source = np.asarray(lethal_grid) >= 99
    out = np.zeros(source.shape, dtype=np.uint8)
    if not source.any():
        return out
    if cv2 is not None:
        free_space = (~source).astype(np.uint8)
        distances = cv2.distanceTransform(free_space, cv2.DIST_L2,
                                           cv2.DIST_MASK_PRECISE) * resolution
        active = distances <= radius_m
        out[active] = np.rint(98 * np.exp(-decay * distances[active])).astype(np.uint8)
        out[source] = 100
        return out
    distances = np.full(source.shape, np.inf, dtype=np.float32)
    queue = []
    for row, col in np.argwhere(source):
        distances[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))
    height, width = source.shape
    neighbors = ((-1, 0, resolution), (1, 0, resolution),
                 (0, -1, resolution), (0, 1, resolution),
                 (-1, -1, resolution * math.sqrt(2)),
                 (-1, 1, resolution * math.sqrt(2)),
                 (1, -1, resolution * math.sqrt(2)),
                 (1, 1, resolution * math.sqrt(2)))
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > distances[row, col] or distance > radius_m:
            continue
        for dr, dc, step in neighbors:
            nr, nc = row + dr, col + dc
            candidate = distance + step
            if (0 <= nr < height and 0 <= nc < width and candidate <= radius_m and
                    candidate < distances[nr, nc]):
                distances[nr, nc] = candidate
                heapq.heappush(queue, (candidate, nr, nc))
    active = np.isfinite(distances) & (distances <= radius_m)
    out[active] = np.rint(98 * np.exp(-decay * distances[active])).astype(np.uint8)
    out[source] = 100
    return out


class TemporalVoxelGrid:
    """2.5-D temporal occupancy store projected to a costmap."""

    def __init__(self, spec: GridSpec, z_bins=16, z_min=-1.0, z_max=3.0,
                 persistence_s=2.0):
        self.spec = spec
        self.z_bins = z_bins
        self.z_min = z_min
        self.z_max = z_max
        self.persistence_s = persistence_s
        self.last_seen = np.full((*spec.shape, z_bins), -np.inf, dtype=np.float64)

    def observe(self, points_xyz: np.ndarray, stamp_s: float) -> None:
        pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
        h, w = self.spec.shape
        ix = np.floor((pts[:, 0] + self.spec.width_m / 2) / self.spec.resolution).astype(int)
        iy = np.floor((pts[:, 1] + self.spec.height_m / 2) / self.spec.resolution).astype(int)
        iz = np.floor((pts[:, 2] - self.z_min) / (self.z_max - self.z_min) * self.z_bins).astype(int)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (iz >= 0) & (iz < self.z_bins)
        self.last_seen[iy[valid], ix[valid], iz[valid]] = stamp_s

    def project(self, stamp_s: float, minimum_voxels=1) -> np.ndarray:
        active = (stamp_s - self.last_seen) <= self.persistence_s
        return np.where(active.sum(axis=2) >= minimum_voxels, 100, 0).astype(np.uint8)


def rasterize_predictions(spec: GridSpec, tracks, horizons=(0.5, 1.0, 2.0, 3.0),
                          radius_m=1.2) -> np.ndarray:
    """Rasterize constant-velocity (x, y, vx, vy) tracks with horizon decay."""
    grid = np.zeros(spec.shape, dtype=np.uint8)
    h, w = spec.shape
    r = int(math.ceil(radius_m / spec.resolution))
    for x, y, vx, vy in tracks:
        for horizon in horizons:
            px, py = x + vx * horizon, y + vy * horizon
            cx = int((px + spec.width_m / 2) / spec.resolution)
            cy = int((py + spec.height_m / 2) / spec.resolution)
            cost = max(25, round(100 - 15 * horizon))
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r and 0 <= cx + dx < w and 0 <= cy + dy < h:
                        grid[cy + dy, cx + dx] = max(grid[cy + dy, cx + dx], cost)
    return grid
