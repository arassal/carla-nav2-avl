#!/usr/bin/env python3
"""Repeatable dependency-light timing smoke test; results are not vehicle claims."""

import time
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from zedx_vision_costmap.core import GridSpec, inflate
from zedx_vision_costmap.perception import (
    observed_mask_from_rays, Pose2D, WorldOccupancyModel,
)


def timed(label, function):
    started = time.perf_counter()
    result = function()
    elapsed_ms = (time.perf_counter() - started) * 1000
    print(f'{label}: {elapsed_ms:.1f} ms')
    return result


def main():
    rng = np.random.default_rng(7)
    spec = GridSpec()
    source = np.zeros(spec.shape, dtype=np.uint8)
    rows = rng.integers(0, spec.shape[0], 250)
    cols = rng.integers(0, spec.shape[1], 250)
    source[rows, cols] = 100
    inflated = timed('inflate_60m_grid_250_sources',
                     lambda: inflate(source, 2.5, spec.resolution, 1.2))

    points = np.column_stack((rng.uniform(0.5, 25.0, 15000),
                              rng.uniform(-12.0, 12.0, 15000),
                              rng.uniform(-1.0, 2.5, 15000)))
    origins = np.zeros_like(points)
    observed = timed('camera_visibility_1200_rays',
                     lambda: observed_mask_from_rays(
                         spec, points, origins, max_rays=1200, dilation_cells=1))
    model = WorldOccupancyModel(spec)
    timed('world_voxel_observe_15000_points',
          lambda: model.observe(points, Pose2D(), origins, 1.0))
    static, temporal = timed('world_voxel_project', lambda: model.project(Pose2D(), 1.0))
    print(f'inflated_cells: {int((inflated > 0).sum())}')
    print(f'observed_cells: {int(observed.sum())}')
    print(f'active_voxels: {len(model.last_seen)}')
    print(f'projected_cells: {int(((static > 0) | (temporal > 0)).sum())}')


if __name__ == '__main__':
    main()
