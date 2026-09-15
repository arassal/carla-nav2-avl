import numpy as np
import pytest

from perception_costmap.costmap_cloud import raycast_costmap


def _rays():
    ray_x = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])
    ray_y = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    return ray_x, ray_y


def test_raycast_keeps_nearest_threshold_hit_and_clears_empty_ray():
    grid = np.zeros((2, 3), np.int8)
    grid[0, 1:] = 97
    ray_x, ray_y = _rays()
    points, hits = raycast_costmap(
        grid, 1.0, 0.0, 0.0, ray_x, ray_y,
        np.array([4.0, 4.0]), np.array([0.5, 1.5]), 97, 0.35)
    assert np.array_equal(hits, [True, False])
    assert np.allclose(points[0], [1.5, 0.5, 0.35])
    assert np.allclose(points[1], [4.0, 1.5, 0.35])


def test_raycast_does_not_promote_cost_below_threshold():
    grid = np.full((2, 3), 96, np.int8)
    ray_x, ray_y = _rays()
    _, hits = raycast_costmap(
        grid, 1.0, 0.0, 0.0, ray_x, ray_y,
        np.array([4.0, 4.0]), np.array([0.5, 1.5]), 97, 0.35)
    assert not hits.any()


def test_raycast_rejects_invalid_threshold():
    ray_x, ray_y = _rays()
    with pytest.raises(ValueError):
        raycast_costmap(
            np.zeros((2, 3)), 1.0, 0.0, 0.0, ray_x, ray_y,
            np.array([4.0, 4.0]), np.array([0.5, 1.5]), 0, 0.35)
