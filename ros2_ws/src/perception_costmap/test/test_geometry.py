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


def test_world_to_cell_rejects_points_just_below_the_grid_min():
    """The method points_to_grid_mask was named after had the same bug.

    int() truncates toward zero, so anything in (min - resolution, min) landed
    on index 0 and passed the bounds check. The vectorized caller above was
    fixed with np.floor; world_to_cell kept truncating until 2026-09-10.
    Only the lower edge was ever affected.
    """
    g = GridSpec(x_min=-4.0, x_max=16.0, y_min=-10.0, y_max=10.0,
                 resolution=0.1)
    assert g.world_to_cell(-4.05, 0.0) is None      # was (0, 100)
    assert g.world_to_cell(-4.099, 0.0) is None     # was (0, 100)
    assert g.world_to_cell(0.0, -10.05) is None     # was (40, 0)
    assert g.world_to_cell(0.0, -10.5) is None      # was already None

    # the exact lower corner is INSIDE, and the upper edge still excludes
    assert g.world_to_cell(-4.0, -10.0) == (0, 0)
    assert g.world_to_cell(15.99, 9.99) == (199, 199)
    assert g.world_to_cell(16.0, 0.0) is None
    assert g.world_to_cell(0.0, 10.0) is None


def test_world_to_cell_agrees_with_points_to_grid_mask():
    """The two must bin identically -- they are the same operation.

    They disagreed for a year on points just below the grid minimum, which is
    exactly the kind of drift a shared fixture catches and two separate
    implementations do not.
    """
    g = GridSpec(x_min=-4.0, x_max=16.0, y_min=-10.0, y_max=10.0,
                 resolution=0.1)
    probes = [(-4.05, 0.0), (-4.0, -10.0), (0.0, 0.0), (5.0, 2.5),
              (15.99, 9.99), (16.0, 0.0), (0.0, -10.05), (-3.999, 9.999)]
    for x, y in probes:
        cell = g.world_to_cell(x, y)
        mask = points_to_grid_mask(np.array([[x, y, 1.0]]), g)
        if cell is None:
            assert not mask.any(), "%r: world_to_cell dropped it, mask kept it" % ((x, y),)
        else:
            col, row = cell
            assert mask[row, col], "%r: disagreed on the cell" % ((x, y),)
            assert mask.sum() == 1


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


def test_optical_depth_point_transforms_to_forward_robot_axis():
    point = bev.optical_points_to_robot(
        np.array([[0.0, 0.0, 5.0]]), (0.5, 0.0, 1.0), 0.0, 0.0)[0]
    assert np.allclose(point, [5.5, 0.0, 1.0])


def test_draw_grid_overlay_changes_pixels_and_preserves_input():
    from perception_costmap.bev import draw_grid_on_image
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    img_pts = [(0, 200), (640, 200), (640, 360), (0, 360)]
    wld_pts = [(10, 5), (10, -5), (2, -2), (2, 2)]
    H = bev.homography_from_points(img_pts, wld_pts, g)
    img = np.zeros((360, 640, 3), np.uint8)
    out = draw_grid_on_image(img, H, g)
    assert out.shape == img.shape
    assert out.any()                 # lines were drawn
    assert not img.any()             # input untouched


def test_unknown_cost_makes_blind_cells_traversable():
    """unknown_cost > 0: blind cells get the mild penalty, not -1."""
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)  # 2x2
    road = np.zeros((2, 2), bool)
    obst = np.zeros((2, 2), bool)
    known = np.zeros((2, 2), bool)
    known[0, 0] = True
    known[0, 1] = True
    road[0, 0] = True
    cost = build_cost_array(g, road, obst, known_mask=known, unknown_cost=25)
    assert cost[0, 0] == FREE          # observed road
    assert cost[1, 1] == 25            # blind -> traversable w/ penalty
    assert cost[0, 1] == 65            # observed off-road default


def test_unknown_cost_default_keeps_ros_unknown():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    empty = np.zeros((2, 2), bool)
    known = np.zeros((2, 2), bool)
    cost = build_cost_array(g, empty, empty, known_mask=known)
    assert (cost == UNKNOWN).all()


def test_obstacle_inflation_bleeds_into_blind_cells():
    """A blind cell adjacent to a detected obstacle must cost MORE than the
    plain unknown penalty -- the 'safe unless obstacles nearby' rule."""
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    shape = (g.height, g.width)
    road = np.zeros(shape, bool)
    obst = np.zeros(shape, bool)
    known = np.zeros(shape, bool)   # everything blind
    obst[50, 50] = True             # except one detected obstacle
    cost = build_cost_array(g, road, obst, known_mask=known,
                            unknown_cost=25, inflation_radius=0.8,
                            cost_scaling_factor=4.0)
    assert cost[50, 50] == LETHAL
    assert cost[50, 52] > 25        # blind cell 0.2m away: inflated
    assert cost[50, 51] > cost[50, 53]   # decays with distance
    assert cost[50, 90] == 25       # far blind cell: plain penalty


def test_semantic_exclusion_radius_survives_high_cost_bridge():
    g = GridSpec(x_min=0, x_max=6, y_min=0, y_max=4, resolution=0.1)
    shape = (g.height, g.width)
    road = np.ones(shape, bool)
    known = np.ones(shape, bool)
    person = np.zeros(shape, bool)
    vehicle = np.zeros(shape, bool)
    person[10, 10] = True
    vehicle[30, 10] = True
    obstacles = person | vehicle
    layers = {
        "person": {
            "mask": person, "radius": 2.0, "scaling": 1.5,
            "exclusion_radius": 1.0,
        },
        "vehicle": {
            "mask": vehicle, "radius": 1.0, "scaling": 3.0,
            "exclusion_radius": 0.4,
        },
    }

    cost = build_cost_array(
        g, road, obstacles, known_mask=known, obstacle_layers=layers)

    assert cost[10, 18] == LETHAL   # 0.8 m from person: hard exclusion
    assert 0 < cost[30, 18] < 97    # same distance from vehicle: soft only
    assert cost[30, 13] == LETHAL   # 0.3 m from vehicle: hard exclusion


def test_semantic_obstacle_layer_shape_mismatch_raises():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)
    shape = (g.height, g.width)
    with pytest.raises(ValueError):
        build_cost_array(
            g, np.ones(shape, bool), np.zeros(shape, bool),
            obstacle_layers={"bad": {
                "mask": np.ones((3, 3), bool), "radius": 1.0,
                "scaling": 1.0, "exclusion_radius": 0.5,
            }})
