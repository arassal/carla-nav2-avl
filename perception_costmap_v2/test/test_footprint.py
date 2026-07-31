import numpy as np
from perception_costmap.obstacles import (
    boxes_to_footprint_mask, points_to_grid_mask, remove_ground_plane,
    detect_obstacles_classical, camera_obstacle_mask_to_grid,
)
from perception_costmap.occupancy import GridSpec
from perception_costmap.bev import homography_from_extrinsics, bev_known_mask


def test_footprint_mask_only_bottom_strip():
    # a tall box (roof at y=10, ground at y=100) in a 200x200 image
    box = [[40, 10, 60, 100]]
    mask = boxes_to_footprint_mask(box, (200, 200), footprint_frac=0.25)
    # bottom strip (last 25% of the box height = y in [77.5, 100]) is set
    assert mask[95, 50]
    # the roof area (top of the box) must NOT be marked -- that's the whole
    # point: warping it through IPM would smear the obstacle far away
    assert not mask[15, 50]


def test_footprint_mask_clips_to_image_bounds():
    box = [[-10, -10, 20, 20]]
    mask = boxes_to_footprint_mask(box, (50, 50), footprint_frac=0.5)
    assert mask.shape == (50, 50)
    assert mask.any()


def test_footprint_mask_empty_boxes():
    mask = boxes_to_footprint_mask(np.zeros((0, 4)), (50, 50))
    assert not mask.any()


def test_footprint_mask_degenerate_box_ignored():
    # x2 < x1 after clipping -> skipped, no crash
    box = [[100, 100, 90, 90]]
    mask = boxes_to_footprint_mask(box, (50, 50))
    assert not mask.any()


def test_remove_ground_plane_filters_by_z_band():
    pts = np.array([
        [1.0, 0.0, -1.0],   # below band -> ground, dropped
        [1.0, 0.0, 0.5],    # in band -> kept
        [1.0, 0.0, 5.0],    # above band -> dropped
    ])
    out = remove_ground_plane(pts, z_min=-0.3, z_max=3.0)
    assert out.shape == (1, 3)
    assert out[0, 2] == 0.5


def test_default_z_min_rejects_actual_road_returns():
    """The band test above uses z=-1.0 as its "ground" point, which no real
    lidar produces: points arrive with the mount height already added, so
    road returns sit at z ~= 0. That gap let a negative default z_min ship,
    which kept the entire road surface and published it LETHAL. Pin the
    default against realistic road noise instead."""
    road = np.column_stack((
        np.linspace(2.0, 16.0, 50),
        np.zeros(50),
        np.random.RandomState(0).normal(0.0, 0.02, 50),   # +/- a few cm
    ))
    assert remove_ground_plane(road).shape[0] == 0

    # ...while anything that matters is still kept: a pedestrian's torso
    person = np.array([[6.0, 0.0, 0.9], [6.0, 0.0, 1.5]])
    assert remove_ground_plane(person).shape[0] == 2


def test_remove_ground_plane_empty_input():
    out = remove_ground_plane(np.zeros((0, 3)))
    assert out.shape == (0, 3)


def test_classical_detector_finds_high_contrast_blob():
    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    img[40:60, 40:60] = 0    # a sharp dark square against a bright field
    mask = detect_obstacles_classical(img, thresh=30, min_area=50)
    assert mask[50, 50]
    assert not mask[5, 5]


def test_classical_detector_uniform_image_no_obstacles():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = detect_obstacles_classical(img)
    assert not mask.any()


# ---------- camera_obstacle_mask_to_grid known_mask clipping ----------
# Regression test for a real bug found while building the synthetic demo:
# warping a camera obstacle mask through IPM without clipping to the
# camera's own known-footprint can plant a spurious obstacle cell via a
# "mirror cell" (a destination grid cell whose inverse-mapped source ray
# has near-zero/negative projective depth -- see bev.bev_known_mask's
# docstring for the full explanation). A mask that is entirely inside the
# image must, after clipping, only ever mark cells inside known_mask.

def test_camera_obstacle_mask_to_grid_clips_to_known_mask():
    grid = GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10, resolution=0.2)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H = homography_from_extrinsics(K, (0, 0, 1.5), pitch_deg=15, yaw_deg=0.0, grid=grid)
    image_shape = (480, 640)
    known = bev_known_mask(H, image_shape, grid)

    # A full-image "detection" mask -- the worst case for mirror-cell
    # leakage, since every image pixel (including any near the horizon
    # that warps unpredictably) is marked.
    full_mask = np.ones(image_shape, dtype=np.uint8)

    unclipped = camera_obstacle_mask_to_grid(full_mask, H, grid)
    clipped = camera_obstacle_mask_to_grid(full_mask, H, grid, known_mask=known)

    # Clipped result must never contain a True cell outside known_mask.
    assert not (clipped & ~known).any()
    # And every clipped cell really is a subset of the unclipped warp.
    assert not (clipped & ~unclipped).any()


def test_camera_obstacle_mask_to_grid_without_known_mask_is_unclipped():
    grid = GridSpec(x_min=0, x_max=10, y_min=0, y_max=10, resolution=1.0)
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
    H = homography_from_extrinsics(K, (0, 0, 1.5), pitch_deg=15, yaw_deg=0.0, grid=grid)
    mask = np.ones((480, 640), dtype=np.uint8)
    out = camera_obstacle_mask_to_grid(mask, H, grid)
    assert out.dtype == bool
