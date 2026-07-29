"""
bev.py
------
Inverse-perspective mapping (IPM): warp a forward camera image (or a mask
derived from it) into the top-down metric costmap grid defined by a GridSpec.

Two ways to get the homography:

1. ``homography_from_points`` -- give 4 points in the image and their known
   ground positions (metres, robot frame). Works for any camera; calibrate
   once by clicking 4 ground points. This is the robust, recommended path.

2. ``homography_from_camera`` -- compute it analytically from camera
   intrinsics K + mounting (height, pitch). Convenient in CARLA where these
   are known exactly. Verify against a real frame before trusting it.

The homography maps image pixels -> grid cell coords (col, row), so the warped
output lines up cell-for-cell with the OccupancyGrid.
"""

import numpy as np
import cv2

from .occupancy import GridSpec


def _world_to_grid_affine(grid: GridSpec) -> np.ndarray:
    """3x3 affine mapping world ground (x, y, 1) -> grid (col, row, 1)."""
    s = 1.0 / grid.resolution
    return np.array([
        [s, 0.0, -grid.x_min * s],
        [0.0, s, -grid.y_min * s],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def homography_from_points(image_pts, world_pts, grid: GridSpec) -> np.ndarray:
    """
    image_pts: 4x2 pixel coords (u, v).
    world_pts: 4x2 ground coords (x forward, y left) in metres.
    Returns 3x3 homography mapping image pixels -> grid (col, row).
    """
    image_pts = np.asarray(image_pts, dtype=np.float32).reshape(4, 2)
    world_pts = np.asarray(world_pts, dtype=np.float64).reshape(4, 2)

    A = _world_to_grid_affine(grid)
    grid_pts = []
    for (x, y) in world_pts:
        v = A @ np.array([x, y, 1.0])
        grid_pts.append([v[0], v[1]])
    grid_pts = np.asarray(grid_pts, dtype=np.float32)

    # findHomography (SVD), not getPerspectiveTransform: the latter pins
    # h33 = 1 and silently returns a ZERO matrix when the true homography
    # has h33 = 0 (horizon through pixel (0,0) -- the current default
    # placeholder points hit exactly this, found 2026-07-03).
    H, _ = cv2.findHomography(image_pts, grid_pts, method=0)
    if H is None or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("degenerate IPM point correspondence (collinear or "
                         "crossed points?): %r -> %r" % (image_pts.tolist(),
                                                         world_pts.tolist()))
    return H


def homography_from_extrinsics(K, cam_xyz, pitch_deg, yaw_deg, grid: GridSpec) -> np.ndarray:
    """
    Ground-plane homography for a camera mounted at cam_xyz (robot frame,
    metres), pitched down pitch_deg and yawed yaw_deg (CCW, +left; 0 = facing
    +x). This is homography_from_camera generalised for side/rear cameras.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    th = np.radians(pitch_deg)
    yw = np.radians(yaw_deg)

    R0 = np.array([[0.0, -1.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [1.0, 0.0, 0.0]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(th), -np.sin(th)],
                   [0.0, np.sin(th), np.cos(th)]])
    Rz_inv = np.array([[np.cos(yw), np.sin(yw), 0.0],     # world -> cam-heading
                       [-np.sin(yw), np.cos(yw), 0.0],
                       [0.0, 0.0, 1.0]])
    R = Rx @ R0 @ Rz_inv

    C = np.asarray(cam_xyz, dtype=np.float64)
    t = -R @ C
    H_world2img = K @ np.column_stack((R[:, 0], R[:, 1], t))
    H_img2world = np.linalg.inv(H_world2img)
    return _world_to_grid_affine(grid) @ H_img2world


def homography_from_camera(K, cam_height, pitch_deg, grid: GridSpec) -> np.ndarray:
    """
    Analytic ground-plane homography from intrinsics + mounting.

    K: 3x3 camera intrinsics. cam_height: metres above ground.
    pitch_deg: downward tilt of the optical axis (0 = looking at the horizon).

    World/robot frame: x forward, y left, z up. OpenCV camera frame: x right,
    y down, z forward. NOTE: verify the sign/þitch convention against your
    actual camera_info before relying on this in the field.
    """
    return homography_from_extrinsics(K, (0.0, 0.0, cam_height), pitch_deg, 0.0, grid)


def warp_to_bev(image, H, grid: GridSpec, interp=cv2.INTER_NEAREST):
    """Warp an image/mask from camera view to the BEV grid (height, width)."""
    return cv2.warpPerspective(image, H, (grid.width, grid.height), flags=interp)


def bev_known_mask(H, image_shape, grid: GridSpec) -> np.ndarray:
    """
    The set of grid cells the camera actually sees (its ground footprint),
    used as the ``known_mask`` so cells outside the camera's view stay
    UNKNOWN instead of being marked off-road.

    Computed analytically: a cell is known iff its source pixel lies inside
    the image AND has positive projective depth. The previous
    warp-an-all-ones-image approach admitted mirror cells BEHIND the camera
    plane that project through the camera centre onto sky pixels (negative
    w, which warpPerspective happily samples) -- a pitch-0 side camera
    "observed" ground on the opposite side of the car (found 2026-07-03).
    """
    h, w = image_shape[:2]
    try:
        Hinv = np.linalg.inv(H)                    # grid -> image, homogeneous
    except np.linalg.LinAlgError:
        # degenerate H: the camera sees nothing rather than garbage
        return np.zeros((grid.height, grid.width), dtype=bool)
    jj, ii = np.meshgrid(np.arange(grid.width) + 0.5,
                         np.arange(grid.height) + 0.5)
    p = np.tensordot(Hinv, np.stack((jj, ii, np.ones_like(jj))), axes=1)
    # fix the homography's global sign (arbitrary for points-mode H) with a
    # pixel that must see real ground: the image's bottom-centre
    g = H @ np.array([w / 2.0, h - 1.0, 1.0])
    s = np.sign((Hinv @ (g / g[2]))[2]) or 1.0
    wcell = p[2] * s
    with np.errstate(divide="ignore", invalid="ignore"):
        u = p[0] / p[2]
        v = p[1] / p[2]
    return (wcell > 1e-9) & (u >= 0) & (u < w) & (v >= 0) & (v < h)


def draw_grid_on_image(img_bgr, H, grid: GridSpec, spacing_m=1.0):
    """
    Project the metric grid back into the camera image (green lines every
    spacing_m). Human calibration check: stand a marker at a known distance
    and confirm the drawn line lands on it. H maps image->grid, so we draw
    with its inverse.
    """
    out = img_bgr.copy()
    Hinv = np.linalg.inv(H)
    h, w = img_bgr.shape[:2]

    def world_to_px(x, y):
        col = (x - grid.x_min) / grid.resolution
        row = (y - grid.y_min) / grid.resolution
        p = Hinv @ np.array([col, row, 1.0])
        if abs(p[2]) < 1e-9:
            return None
        u, v = p[0] / p[2], p[1] / p[2]
        if -w <= u <= 2 * w and -h <= v <= 2 * h:
            return int(round(u)), int(round(v))
        return None

    for x in np.arange(np.ceil(grid.x_min), grid.x_max + 1e-6, spacing_m):
        pts = [world_to_px(x, y) for y in np.linspace(grid.y_min, grid.y_max, 40)]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 1)
        if pts:
            cv2.putText(out, "%gm" % x, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
    for y in np.arange(np.ceil(grid.y_min), grid.y_max + 1e-6, spacing_m):
        pts = [world_to_px(x, y) for x in np.linspace(max(0.5, grid.x_min), grid.x_max, 40)]
        pts = [p for p in pts if p is not None]
        for a, b in zip(pts, pts[1:]):
            cv2.line(out, a, b, (0, 255, 0), 1)
        if pts:
            # signed label so a left-right-mirrored homography is visible
            cv2.putText(out, "y=%+gm" % y, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 255), 1)
    return out
