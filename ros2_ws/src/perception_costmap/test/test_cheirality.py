"""Offline tests for the mirror-projection (cheirality) fix (no ROS).

A ground-plane homography maps every grid cell to SOME pixel -- including
cells behind the camera plane, which project through the camera centre onto
above-horizon pixels with negative projective depth. warpPerspective samples
them anyway, so before the fix a pitch-0 side camera "observed" ground on the
opposite side of the car (2026-07-03).
"""

import numpy as np

from perception_costmap.occupancy import GridSpec, build_cost_array, FREE, LETHAL, UNKNOWN
from perception_costmap import bev

G = GridSpec(x_min=-4, x_max=16, y_min=-10, y_max=10, resolution=0.1)

# ZED X-ish intrinsics at SVGA (960x600)
K = np.array([[370.0, 0.0, 480.0],
              [0.0, 370.0, 300.0],
              [0.0, 0.0, 1.0]])
IMG_SHAPE = (600, 960)

# dinosaur's left camera: yaw +90, pitch 0, 0.61 m up
LEFT = dict(cam_xyz=(0.098, 0.286, 0.6126), pitch_deg=0.0, yaw_deg=90.0)


def _known_left():
    H = bev.homography_from_extrinsics(K, LEFT["cam_xyz"], LEFT["pitch_deg"],
                                       LEFT["yaw_deg"], G)
    return bev.bev_known_mask(H, IMG_SHAPE, G)


def _cell(x, y):
    col, row = G.world_to_cell(x, y)
    return row, col


def test_side_camera_excludes_mirror_side():
    known = _known_left()
    # cells on the car's RIGHT (-y): the left camera cannot see them
    for x, y in [(0.0, -3.0), (0.0, -8.0), (2.0, -5.0)]:
        r, c = _cell(x, y)
        assert not known[r, c], "left camera claims to see (%s, %s)" % (x, y)


def test_side_camera_keeps_real_footprint():
    known = _known_left()
    # cells on the car's LEFT, straight out from the camera: must stay known
    for x, y in [(0.1, 3.0), (0.1, 6.0)]:
        r, c = _cell(x, y)
        assert known[r, c], "left camera lost real footprint at (%s, %s)" % (x, y)
    # and the mask must be a minority of the grid, not ~half of it
    assert known.mean() < 0.35


def test_front_camera_footprint_sane():
    # pitched-down forward camera: sees ahead, nothing behind
    H = bev.homography_from_extrinsics(K, (0.68, 0.0, 0.45), 15.0, 0.0, G)
    known = bev.bev_known_mask(H, IMG_SHAPE, G)
    r, c = _cell(5.0, 0.0)
    assert known[r, c]
    r, c = _cell(-2.0, 0.0)
    assert not known[r, c]


def test_points_mode_mask_unchanged_in_footprint():
    # points-mode H (arbitrary global sign): the calibration quad interior
    # must remain known -- the sign anchor must not invert it
    img_pts = [[0, 160], [640, 160], [640, 320], [0, 320]]
    world_pts = [[18, 8], [18, -8], [3, -4], [3, 4]]
    H = bev.homography_from_points(img_pts, world_pts, G)
    known = bev.bev_known_mask(H, (480, 640), G)
    r, c = _cell(10.0, 0.0)   # centre of the quad
    assert known[r, c]


def test_points_mode_default_placeholders_not_degenerate():
    # the default placeholder correspondence has h33 = 0:
    # getPerspectiveTransform (which pins h33 = 1) silently returned a ZERO
    # matrix for it. findHomography must return a usable, invertible H that
    # maps the calibration quad correctly.
    img_pts = [[0, 160], [640, 160], [640, 320], [0, 320]]
    world_pts = [[18, 8], [18, -8], [3, -4], [3, 4]]
    H = bev.homography_from_points(img_pts, world_pts, G)
    assert abs(np.linalg.det(H)) > 1e-12
    for (u, v), (x, y) in zip(img_pts, world_pts):
        p = H @ np.array([u, v, 1.0])
        col, row = p[0] / p[2], p[1] / p[2]
        ecol, erow = (x - G.x_min) / G.resolution, (y - G.y_min) / G.resolution
        assert abs(col - ecol) < 0.5 and abs(row - erow) < 0.5


def test_road_outside_known_stays_unknown():
    g = GridSpec(x_min=0, x_max=2, y_min=0, y_max=2, resolution=1.0)  # 2x2
    road = np.zeros((2, 2), bool)
    obst = np.zeros((2, 2), bool)
    known = np.zeros((2, 2), bool)
    known[0, 0] = True
    road[0, 0] = True       # observed road -> FREE
    road[1, 1] = True       # mirror "road" outside known -> must NOT be FREE
    obst[0, 1] = True       # lidar obstacle outside camera known -> LETHAL
    cost = build_cost_array(g, road, obst, known_mask=known)
    assert cost[0, 0] == FREE
    assert cost[1, 1] == UNKNOWN
    assert cost[0, 1] == LETHAL
