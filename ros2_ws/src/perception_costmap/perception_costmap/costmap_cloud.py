"""Pure conversion from a robot-frame costmap to ray endpoints for Nav2."""

import numpy as np


def raycast_costmap(grid, resolution, origin_x, origin_y, ray_x, ray_y,
                    clear_x, clear_y, obstacle_threshold, point_z):
    """Return one hit or clearing endpoint per bearing and the hit mask."""
    values = np.asarray(grid)
    if values.ndim != 2 or ray_x.shape != ray_y.shape:
        raise ValueError("expected a 2D grid and equal ray coordinate shapes")
    if ray_x.ndim != 2 or len(clear_x) != ray_x.shape[0]:
        raise ValueError("clearing endpoints must match the ray count")
    if not 1 <= int(obstacle_threshold) <= 100:
        raise ValueError("obstacle_threshold must be in [1, 100]")

    height, width = values.shape
    ix = np.floor((ray_x - origin_x) / resolution).astype(np.int32)
    iy = np.floor((ray_y - origin_y) / resolution).astype(np.int32)
    inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
    sampled = np.full(ray_x.shape, -1, dtype=np.int16)
    sampled[inside] = values[iy[inside], ix[inside]]

    hits = sampled >= int(obstacle_threshold)
    has_hit = hits.any(axis=1)
    first_hit = hits.argmax(axis=1)
    hit_rows = np.flatnonzero(has_hit)
    points = np.column_stack((clear_x, clear_y,
                              np.full(len(clear_x), point_z))).astype(np.float32)
    if hit_rows.size:
        hit_columns = first_hit[hit_rows]
        points[hit_rows, 0] = ray_x[hit_rows, hit_columns]
        points[hit_rows, 1] = ray_y[hit_rows, hit_columns]
    return points, has_hit
