"""ROS-independent costmap algorithms, intentionally unit-testable without ROS."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Tuple

import numpy as np

LAYER_NAMES = (
    'lanelet', 'static_obstacle', 'spatio_temporal_voxel', 'prediction',
    'inflation', 'traffic_regulation', 'road_condition',
)


@dataclass(frozen=True)
class GridSpec:
    width_m: float = 60.0
    height_m: float = 60.0
    resolution: float = 0.20

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
    """Euclidean inflation with exponential decay; dependency-free reference code."""
    out = np.zeros_like(lethal_grid, dtype=np.uint8)
    occupied = np.argwhere(lethal_grid >= 99)
    radius_cells = int(math.ceil(radius_m / resolution))
    height, width = out.shape
    for oy, ox in occupied:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                distance = math.hypot(dx, dy) * resolution
                if distance > radius_m:
                    continue
                y, x = oy + dy, ox + dx
                if 0 <= y < height and 0 <= x < width:
                    cost = 99 if distance == 0 else round(98 * math.exp(-decay * distance))
                    out[y, x] = max(out[y, x], cost)
    out[lethal_grid >= 99] = 100
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


def infer_road_condition(bgr: np.ndarray) -> Tuple[str, float, int]:
    """Deterministic baseline classifier for CARLA validation, not a safety model."""
    image = np.asarray(bgr, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('expected HxWx3 BGR image')
    lower = image[image.shape[0] // 2:]
    brightness = float(lower.mean())
    channel_spread = float(np.mean(np.max(lower, axis=2) - np.min(lower, axis=2)))
    gray = lower.mean(axis=2)
    contrast = float(gray.std())
    if brightness < 45:
        return 'low_visibility', min(1.0, (55 - brightness) / 40), 45
    if channel_spread < 12 and contrast > 35:
        return 'wet', min(1.0, contrast / 70), 35
    if brightness > 190 and channel_spread < 18:
        return 'snow_or_glare', min(1.0, brightness / 255), 55
    return 'dry', 0.70, 0
