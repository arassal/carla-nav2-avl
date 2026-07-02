"""Offline tests for the costmap geometry core (no ROS required)."""

import numpy as np
import pytest

from perception_costmap.occupancy import (
    GridSpec, build_cost_array, UNKNOWN, FREE, LETHAL,
)
from perception_costmap import bev
from perception_costmap.obstacles import points_to_grid_mask


def test_gridspec_dimensions():
    g = GridSpec(x_min=-4, x_max=16, y_min=-10, y_max=10, resolution=0.1)
    assert g.width == 200    # 20 m of x at 0.1 m
    assert g.height == 200   # 20 m of y at 0.1 m


def test_world_to_cell_roundtrip():
    g = GridSpec(resolution=0.1)
    # robot origin (0,0) should land inside the grid
    cell = g.world_to_cell(0.0, 0.0)
    assert cell is not None
    col, row = cell
    x, y = g.cell_to_world(col, row)
    assert abs(x - 0.0) <= g.resolution
    assert abs(y - 0.0) <= g.resolution
    # a point well outside the extent returns None
    assert g.world_to_cell(1000.0, 0.0) is None


def test_cost_priority_and_values():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)  # 2x2
    road = np.zeros((g.height, g.width), bool)
    obst = np.zeros((g.height, g.width), bool)
    known = np.zeros((g.height, g.width), bool)

    known[:, :] = True          # everything observed
    road[0, 0] = True           # one road cell
    road[1, 1] = True
    obst[1, 1] = True           # obstacle overrides road on the same cell

    cost = build_cost_array(g, road, obst, known_mask=known, offroad_cost=LETHAL)
    assert cost[0, 0] == FREE          # road
    assert cost[1, 1] == LETHAL        # obstacle wins over road
    assert cost[0, 1] == LETHAL        # observed, off-road -> lethal


def test_unobserved_cells_are_unknown():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    road = np.zeros((g.height, g.width), bool)
    obst = np.zeros((g.height, g.width), bool)
    known = np.zeros((g.height, g.width), bool)
    known[0, 0] = True
    road[0, 0] = True
    cost = build_cost_array(g, road, obst, known_mask=known)
    assert cost[0, 0] == FREE
    assert cost[1, 1] == UNKNOWN       # never observed


def test_mask_shape_mismatch_raises():
    g = GridSpec(resolution=0.1)
    bad = np.zeros((10, 10), bool)
    with pytest.raises(ValueError):
        build_cost_array(g, bad, bad)


def test_ipm_homography_maps_ground_rect_into_grid():
    """A known ground rectangle, warped via a synthetic 'camera', should come
    back to the right cells through the 4-point homography."""
    g = GridSpec(x_min=0, x_max=20, y_min=-10, y_max=10, resolution=0.1)

    # Four ground points (x fwd, y left) and made-up image pixels for them.
    world_pts = [(2.0, -3.0), (2.0, 3.0), (18.0, 3.0), (18.0, -3.0)]
    image_pts = [(200, 470), (440, 470), (380, 120), (260, 120)]
    H = bev.homography_from_points(image_pts, world_pts, g)

    # Map each image point through H and confirm it lands on the expected cell.
    for (u, v), (x, y) in zip(image_pts, world_pts):
        p = H @ np.array([u, v, 1.0])
        col, row = p[0] / p[2], p[1] / p[2]
        exp_col = (x - g.x_min) / g.resolution
        exp_row = (y - g.y_min) / g.resolution
        assert abs(col - exp_col) < 1.0
        assert abs(row - exp_row) < 1.0


def test_known_mask_is_subset_of_grid():
    g = GridSpec(resolution=0.1)
    world_pts = [(2.0, -3.0), (2.0, 3.0), (18.0, 3.0), (18.0, -3.0)]
    image_pts = [(200, 470), (440, 470), (380, 120), (260, 120)]
    H = bev.homography_from_points(image_pts, world_pts, g)
    known = bev.bev_known_mask(H, (480, 640), g)
    assert known.shape == (g.height, g.width)
    assert known.any()                 # camera sees *something*
    assert not known.all()             # but not the whole grid


def test_points_to_grid_mask_vectorized_matches_cells():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    pts = np.array([[0.5, 0.5, 1.0],    # cell (col 0, row 0)
                    [1.5, 0.5, 1.0],    # cell (col 1, row 0)
                    [9.0, 9.0, 1.0]])   # out of grid, dropped
    m = points_to_grid_mask(pts, g)
    assert m.shape == (2, 2)
    assert m[0, 0] and m[0, 1]
    assert m.sum() == 2


def test_points_to_grid_mask_empty():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    m = points_to_grid_mask(np.zeros((0, 3)), g)
    assert m.shape == (2, 2) and not m.any()


def test_points_just_below_grid_min_are_dropped():
    # the old world_to_cell-based loop truncated toward zero, wrongly binning
    # points in (min - resolution, min) into border cells; floor drops them
    g = GridSpec(x_min=-4.0, x_max=16.0, y_min=-10.0, y_max=10.0, resolution=0.1)
    pts = np.array([[-4.05, 0.0, 1.0],     # just behind the rear edge
                    [0.0, -10.05, 1.0]])   # just right of the right edge
    m = points_to_grid_mask(pts, g)
    assert not m.any()


from perception_costmap.bev import homography_from_extrinsics


def _px_to_world(H, u, v, grid):
    p = H @ np.array([u, v, 1.0])
    col, row = p[0] / p[2], p[1] / p[2]
    return (grid.x_min + col * grid.resolution,
            grid.y_min + row * grid.resolution)


def test_yawed_camera_rotates_ground_points():
    g = GridSpec(x_min=-20, x_max=20, y_min=-20, y_max=20, resolution=0.1)
    K = np.array([[300.0, 0, 320], [0, 300.0, 180], [0, 0, 1]])
    H_fwd = homography_from_extrinsics(K, (0, 0, 1.6), 10.0, 0.0, g)
    H_left = homography_from_extrinsics(K, (0, 0, 1.6), 10.0, 90.0, g)
    u, v = 320.0, 260.0                     # a pixel below the horizon
    xf, yf = _px_to_world(H_fwd, u, v, g)
    xl, yl = _px_to_world(H_left, u, v, g)
    # rotating the camera +90 deg (left) maps (x, y) -> (-y, x)
    assert abs(xl - (-yf)) < 0.05 and abs(yl - xf) < 0.05
