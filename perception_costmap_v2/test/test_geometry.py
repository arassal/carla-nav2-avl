import numpy as np
import pytest

from perception_costmap.occupancy import (
    GridSpec, build_cost_array, inflate_costs, infill_unknown,
    UNKNOWN, FREE, LETHAL, DEFAULT_OFFROAD_COST,
)
from perception_costmap.bev import (
    homography_from_points, homography_from_extrinsics, homography_from_camera,
    warp_to_bev, bev_known_mask,
)
from perception_costmap.obstacles import points_to_grid_mask


# ---------- GridSpec ----------

def test_gridspec_dims():
    g = GridSpec(x_min=-4, x_max=16, y_min=-10, y_max=10, resolution=0.1)
    assert g.width == 200
    assert g.height == 200


def test_gridspec_world_to_cell_roundtrip():
    g = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    cell = g.world_to_cell(3.4, 7.8)
    assert cell == (3, 7)
    x, y = g.cell_to_world(*cell)
    assert x == pytest.approx(3.5)
    assert y == pytest.approx(7.5)


def test_gridspec_world_to_cell_out_of_bounds():
    g = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    assert g.world_to_cell(-1, 5) is None
    assert g.world_to_cell(20, 5) is None


# ---------- points_to_grid_mask (vectorized lidar binning) ----------

def test_points_to_grid_mask_matches_cells():
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


# ---------- inflate_costs ----------

def test_inflate_costs_no_obstacles_is_sentinel():
    mask = np.zeros((10, 10), dtype=bool)
    out = inflate_costs(mask, resolution=0.1)
    assert (out == -1.0).all()


def test_inflate_costs_peak_at_obstacle_decays_with_distance():
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    out = inflate_costs(mask, resolution=0.1, inflation_radius=1.0, cost_scaling_factor=4.0)
    assert out[10, 10] == LETHAL
    # a cell 5 px (0.5m) away should have decayed but still be within radius
    assert 0 < out[10, 15] < LETHAL
    # a cell far beyond the inflation radius is untouched (-1 sentinel)
    assert out[0, 0] == -1.0
    # monotonic decay outward along a ray
    ray = out[10, 10:21]
    ray = ray[ray >= 0]
    assert all(a >= b for a, b in zip(ray, ray[1:]))


# ---------- build_cost_array ----------

def test_build_cost_array_road_is_free():
    g = GridSpec(x_min=0, x_max=5, y_min=0, y_max=5, resolution=1.0)
    shape = (g.height, g.width)
    road = np.ones(shape, dtype=bool)
    obstacle = np.zeros(shape, dtype=bool)
    cost = build_cost_array(g, road, obstacle)
    assert (cost == FREE).all()


def test_build_cost_array_obstacle_is_lethal_and_halos():
    g = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    shape = (g.height, g.width)
    road = np.ones(shape, dtype=bool)
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[5, 5] = True
    cost = build_cost_array(g, road, obstacle, inflation_radius=2.0, cost_scaling_factor=1.0)
    assert cost[5, 5] == LETHAL
    assert 0 < cost[5, 6] < LETHAL          # halo bleeds into free road
    assert cost[0, 0] == FREE               # far cell untouched


def test_build_cost_array_offroad_distinct_from_lethal():
    g = GridSpec(x_min=0, x_max=5, y_min=0, y_max=5, resolution=1.0)
    shape = (g.height, g.width)
    road = np.zeros(shape, dtype=bool)
    road[2, 2] = True
    obstacle = np.zeros(shape, dtype=bool)
    cost = build_cost_array(g, road, obstacle)
    assert cost[2, 2] == FREE
    off = cost[0, 0]
    assert off == DEFAULT_OFFROAD_COST
    assert 0 < off < LETHAL


def test_build_cost_array_unknown_outside_known_mask():
    g = GridSpec(x_min=0, x_max=5, y_min=0, y_max=5, resolution=1.0)
    shape = (g.height, g.width)
    road = np.ones(shape, dtype=bool)      # claims everything is road...
    obstacle = np.zeros(shape, dtype=bool)
    known = np.zeros(shape, dtype=bool)
    known[0, 0] = True                     # ...but only one cell observed
    cost = build_cost_array(g, road, obstacle, known_mask=known)
    assert cost[0, 0] == FREE
    assert cost[1, 1] == UNKNOWN            # unobserved road claim ignored


def test_build_cost_array_road_edge_ramp_is_monotonic_and_bounded():
    g = GridSpec(x_min=0, x_max=20, y_min=0, y_max=1, resolution=1.0)
    shape = (g.height, g.width)
    # road for x in [0,10), off-road for x in [10,20)
    road = np.zeros(shape, dtype=bool)
    road[:, :10] = True
    obstacle = np.zeros(shape, dtype=bool)
    cost = build_cost_array(g, road, obstacle, road_edge_radius=5.0, road_edge_scaling=1.0)
    row = cost[0, :10]
    # increasing toward the boundary (index 9 is right at the edge)
    assert all(a <= b for a, b in zip(row, row[1:]))
    assert row[-1] <= DEFAULT_OFFROAD_COST
    # never confused with a real obstacle
    assert row.max() < LETHAL


def test_build_cost_array_obstacle_layers_per_class_worst_case():
    g = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    shape = (g.height, g.width)
    road = np.ones(shape, dtype=bool)
    person = np.zeros(shape, dtype=bool)
    person[5, 5] = True
    vehicle = np.zeros(shape, dtype=bool)
    vehicle[5, 5] = True  # overlapping halos -> np.maximum takes worst case
    layers = {
        "person": dict(mask=person, radius=3.0, scaling=1.0),
        "vehicle": dict(mask=vehicle, radius=0.5, scaling=10.0),
    }
    cost = build_cost_array(g, road, np.zeros(shape, dtype=bool), obstacle_layers=layers)
    # person's wide/slow halo should reach further than vehicle's tight one
    assert cost[5, 8] > FREE


# ---------- infill_unknown ----------

def test_infill_unknown_seam_between_two_known_road_patches():
    shape = (1, 21)
    cost = np.full(shape, -1.0)
    known = np.zeros(shape, dtype=bool)
    cost[0, :5] = FREE
    cost[0, 16:] = FREE
    known[0, :5] = True
    known[0, 16:] = True
    filled = infill_unknown(cost, known, resolution=0.1, prior=25.0, falloff_m=2.0)
    # right next to the known road on either side, the guess should be
    # close to FREE, not the neutral prior
    assert filled[0, 5] < 25.0
    assert filled[0, 15] < 25.0
    # deep in the middle of a wide blind gap the guess decays toward prior
    assert abs(filled[0, 10] - 25.0) < abs(filled[0, 5] - 25.0)


def test_infill_unknown_noop_when_nothing_known():
    shape = (5, 5)
    cost = np.full(shape, -1.0)
    known = np.zeros(shape, dtype=bool)
    out = infill_unknown(cost, known, resolution=0.1)
    assert np.array_equal(out, cost)


# ---------- BEV homography ----------

def test_homography_from_points_forward_camera_roundtrip():
    g = GridSpec(x_min=0, x_max=20, y_min=-10, y_max=10, resolution=0.1)
    # a simple synthetic forward-looking camera correspondence: image bottom
    # maps to near ground, image top-ish maps to far ground.
    image_pts = [[100, 480], [540, 480], [220, 300], [420, 300]]
    world_pts = [[2.0, -1.5], [2.0, 1.5], [8.0, -1.5], [8.0, 1.5]]
    H = homography_from_points(image_pts, world_pts, g)
    assert H.shape == (3, 3)
    assert abs(np.linalg.det(H)) > 1e-12


def test_homography_from_points_degenerate_raises():
    g = GridSpec(x_min=0, x_max=20, y_min=-10, y_max=10, resolution=0.1)
    image_pts = [[0, 0], [10, 0], [20, 0], [30, 0]]  # collinear
    world_pts = [[1, 0], [2, 0], [3, 0], [4, 0]]
    with pytest.raises(ValueError):
        homography_from_points(image_pts, world_pts, g)


def test_homography_from_extrinsics_yaw_rotates_scene():
    """A camera yawed +90 (looking left) should see the same ground layout
    as a forward camera, just rotated 90deg in the world -- i.e. warping a
    uniform 'ground' test image with both homographies should each produce
    a plausible (non-degenerate) known-footprint region."""
    g = GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10, resolution=0.2)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H_fwd = homography_from_extrinsics(K, (0, 0, 1.5), pitch_deg=15, yaw_deg=0.0, grid=g)
    H_left = homography_from_extrinsics(K, (0, 0, 1.5), pitch_deg=15, yaw_deg=90.0, grid=g)
    img_shape = (480, 640)
    known_fwd = bev_known_mask(H_fwd, img_shape, g)
    known_left = bev_known_mask(H_left, img_shape, g)
    assert known_fwd.any() and known_left.any()
    # forward camera's footprint should be centred toward +x, left camera's
    # footprint toward +y (since it's yawed 90 deg CCW from forward)
    ys_fwd, xs_fwd = np.nonzero(known_fwd)
    ys_left, xs_left = np.nonzero(known_left)
    cx_fwd = g.x_min + (xs_fwd.mean() + 0.5) * g.resolution
    cy_left = g.y_min + (ys_left.mean() + 0.5) * g.resolution
    assert cx_fwd > 0       # forward camera looks toward +x
    assert cy_left > 0      # yawed-left camera looks toward +y


def test_homography_from_camera_is_zero_yaw_special_case():
    g = GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10, resolution=0.2)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H1 = homography_from_camera(K, cam_height=1.5, pitch_deg=15, grid=g)
    H2 = homography_from_extrinsics(K, (0, 0, 1.5), pitch_deg=15, yaw_deg=0.0, grid=g)
    assert np.allclose(H1, H2)


def test_bev_known_mask_rejects_mirror_cells_behind_camera():
    """A pitch=0 side-on camera must not report ground on the opposite side
    of the vehicle as 'known' (the negative-projective-depth bug)."""
    g = GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10, resolution=0.2)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H = homography_from_extrinsics(K, (0, 0, 1.0), pitch_deg=0.0, yaw_deg=90.0, grid=g)
    known = bev_known_mask(H, (480, 640), g)
    ys, xs = np.nonzero(known)
    if len(ys):
        world_y = g.y_min + (ys + 0.5) * g.resolution
        # every known cell should be on the side the camera is yawed toward
        assert (world_y > -1e-6).all() or (world_y < 1e-6).all()


def test_warp_to_bev_shape():
    g = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H = homography_from_camera(K, cam_height=1.5, pitch_deg=20, grid=g)
    img = np.ones((480, 640), dtype=np.uint8) * 255
    warped = warp_to_bev(img, H, g)
    assert warped.shape == (g.height, g.width)
