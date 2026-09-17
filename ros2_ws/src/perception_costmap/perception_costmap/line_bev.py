"""
line_bev.py
-----------
White-line and pothole detection performed in the BIRD'S-EYE VIEW, where a
course line has a CONSTANT width, instead of in the perspective image where
it does not.

Why this exists
===============
IGVC 2026 AutoNav runs on asphalt with boundary lines ~3 in (7.6 cm) wide
taped on the pavement (rules S II.2). In the raw camera frame that line spans
many pixels close to the vehicle and under one pixel far away, so every
image-space size gate -- minimum area, minimum elongation -- is implicitly a
range gate: it throws away distant line segments while admitting nearby bright
blobs. Warping first and detecting second removes the range dependence
entirely: at a known metres-per-pixel, 7.6 cm is the same number of pixels
everywhere in the image.

That turns the line width into a usable prior. A top-hat morphological filter
with a structuring element wider than the line passes structures THINNER than
the element and suppresses everything broader, so a wall, the sky, a white
barrel and a building facade produce no response at all -- not because a
threshold rejected them, but because the operator cannot represent them. This
is the standard road-marking extractor, and it is what makes the blunt
``roi_top_frac`` guard (blank the top 45% of every frame) unnecessary.

Potholes use the same machinery at a different scale: a 2 ft (0.61 m) disc is
isolated by a top-hat with an element sized to the disc, then separated from
line fragments by circularity and metric diameter.

Everything here is threshold-light by design. The one intensity decision -- how
strong a top-hat response counts -- is made from the image's own statistics
(median + k*MAD over the observed footprint) with an absolute floor, so an
empty scene yields no detections rather than the top 1% of noise.
"""

import dataclasses

import numpy as np
import cv2

from . import bev
from .occupancy import GridSpec

# IGVC 2026 rules, S II.2: boundary lines "approximately three inches wide".
IGVC_LINE_WIDTH_M = 0.076
# "Simulated potholes of 2 foot diameter, solid white circles".
IGVC_POTHOLE_DIAMETER_M = 0.61


def fine_grid(grid: GridSpec, upsample: int) -> GridSpec:
    """Same metric extent as ``grid``, ``upsample``x finer cells.

    The costmap runs at 0.1 m/cell, but a 7.6 cm line is 0.76 of one cell --
    narrower than the thing meant to represent it. Detection therefore happens
    on a finer grid and is max-pooled back down, so a line that lands astride
    a cell boundary still marks the cell instead of dropping out.
    """
    if int(upsample) < 1:
        raise ValueError("upsample must be >= 1, got %r" % (upsample,))
    return dataclasses.replace(grid, resolution=grid.resolution / int(upsample))


def fine_homography(H, upsample: int):
    """Lift a homography that maps image -> coarse grid cells to one that maps
    image -> fine grid cells.

    ``bev.homography_from_*`` already folds the world->grid affine into H, and
    the fine grid shares the coarse grid's origin and extent, differing only in
    resolution. Coarse cell (c, r) is therefore fine cell (k*c, k*r): a pure
    scale, so no recalibration is needed and the existing calibrated H stays
    the single source of truth.
    """
    k = float(int(upsample))
    S = np.array([[k, 0.0, 0.0],
                  [0.0, k, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return S @ np.asarray(H, dtype=np.float64)


def _maxpool_bool(mask: np.ndarray, k: int) -> np.ndarray:
    """Downsample a boolean mask by k, marking a coarse cell if ANY fine cell
    in it is set. Deliberately not INTER_NEAREST/INTER_AREA: a sub-cell line
    must survive the trip down to the costmap, not be sampled away."""
    if k == 1:
        return mask
    h, w = mask.shape
    return mask[:h - h % k, :w - w % k].reshape(
        h // k, k, w // k, k).max(axis=(1, 3))


def _robust_threshold(response: np.ndarray, valid: np.ndarray,
                      floor: float, mad_k: float) -> float:
    """median + mad_k*sigma over observed pixels, with an absolute floor.

    The floor is what makes an empty scene return nothing: with no line
    present the MAD is tiny, the statistical term collapses, and the floor
    rejects everything. A percentile threshold would instead always hand back
    its top slice of noise.
    """
    # Subsampled: the statistic is a robust centre/spread over hundreds of
    # thousands of pixels, and every 4th one estimates it just as well for a
    # quarter of the sort cost.
    vals = response[valid][::4]
    if vals.size < 32:
        return float(floor)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(float(floor), med + mad_k * 1.4826 * mad)


def _components(mask_u8):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def _fill(shape, contour):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [contour.reshape(-1, 2)], 255)
    return m > 0


_GEOM_CACHE = {}


def _geometry(H, image_shape, grid, fine, k, max_range_m, erode_k):
    """Footprint + range mask for a camera, cached.

    Both depend only on the homography, the image size and the grid -- none of
    which change while a camera is running -- but recomputing them cost 20 ms
    per frame per camera (bev_known_mask is analytic per cell). They are
    computed once per camera and reused, so a fixed mount pays for them on its
    first frame only.
    """
    key = (np.asarray(H, dtype=np.float64).tobytes(), tuple(image_shape[:2]),
           grid.x_min, grid.y_min, grid.resolution, grid.width, grid.height,
           k, float(max_range_m), int(erode_k))
    hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    known_coarse = bev.bev_known_mask(H, image_shape, grid)
    known_fine = np.repeat(np.repeat(known_coarse, k, axis=0), k, axis=1)
    eroded = cv2.erode(known_fine.astype(np.uint8),
                       np.ones((erode_k, erode_k), np.uint8)) > 0
    valid = eroded & _range_mask(fine, max_range_m)
    if len(_GEOM_CACHE) > 12:          # a handful of cameras, not a leak
        _GEOM_CACHE.clear()
    _GEOM_CACHE[key] = (known_coarse, valid)
    return known_coarse, valid


def _range_mask(fine: GridSpec, max_range_m: float) -> np.ndarray:
    """Cells within max_range_m of the robot origin.

    Beyond that the IPM stretch factor explodes -- a camera 0.47 m off the
    ground spreads one far pixel across metres of ground -- so every feature
    out there is interpolation, not evidence. Bounding the range is cheaper
    and more honest than trying to threshold the smear away.
    """
    xs = np.arange(fine.width) * fine.resolution + fine.x_min
    ys = np.arange(fine.height) * fine.resolution + fine.y_min
    d2 = ys[:, None] ** 2 + xs[None, :] ** 2
    return d2 <= max_range_m * max_range_m


def _shift(mask_u8, dx, dy):
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(mask_u8, M, (mask_u8.shape[1], mask_u8.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0) > 0


def _ridge_contrast(val, comp, line_px, angle_deg, valid):
    """Dark-light-dark test, evaluated on each side SEPARATELY.

    A taped line is bright with darker asphalt on both sides. The edge of a
    large bright object -- a slab, a wall, the lit face of a barrel -- is
    bright on one side only.

    Averaging a ring around the component does NOT catch that: with slab on
    one side and asphalt on the other, the mean lands midway and a
    slab edge scores like a line. (Measured: a 4x4 m slab's outer edge
    survived a ring test at contrast 27, comfortably past an 18 threshold.)
    So each side is sampled independently and the WEAKER one decides.

    Returns the smaller of the two side contrasts; sides that fall outside the
    observed footprint are ignored, and a component with no usable side
    scores 0.
    """
    if comp.sum() < 4:
        return 0.0
    th = np.radians(angle_deg)
    # perpendicular to the component's long axis
    px, py = -np.sin(th), np.cos(th)
    off = max(2.0, 3.0 * line_px)
    comp_u8 = comp.astype(np.uint8)
    inside = float(val[comp].mean())

    contrasts = []
    for sign in (1.0, -1.0):
        side = _shift(comp_u8, sign * off * px, sign * off * py) & valid & ~comp
        if side.sum() >= 8:
            contrasts.append(inside - float(val[side].mean()))
    if not contrasts:
        return 0.0
    return min(contrasts)


def detect_bev(img_bgr, H, grid: GridSpec, upsample=4,
               line_width_m=IGVC_LINE_WIDTH_M,
               min_line_length_m=0.50, max_line_width_m=0.25, min_aspect=3.0,
               response_floor=25.0, mad_k=4.0, white_s_max=90,
               max_range_m=12.0, min_ridge_contrast=18.0,
               pothole_diameter_m=IGVC_POTHOLE_DIAMETER_M,
               pothole_diameter_tol=0.45, min_circularity=0.62,
               want_potholes=True, want_lines=True):
    """Detect course lines and potholes directly in the metric BEV.

    Returns a dict with, in COARSE grid cells ready to OR into a class grid:
        lines, potholes  -- boolean (grid.height, grid.width)
        known            -- camera ground footprint, coarse
    plus diagnostics: line_thr, pothole_thr, n_lines, n_potholes, and the
    fine-resolution masks under lines_fine / potholes_fine for debugging.
    """
    k = int(upsample)
    fine = fine_grid(grid, k)
    Hf = fine_homography(H, k)
    mpp = fine.resolution

    # Convert on the SMALL source frame and warp the two channels we actually
    # use, rather than warping 3 channels and converting 800x800 afterwards.
    hsv_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    val = bev.warp_to_bev(hsv_src[:, :, 2], Hf, fine, interp=cv2.INTER_LINEAR)
    sat = bev.warp_to_bev(hsv_src[:, :, 1], Hf, fine, interp=cv2.INTER_LINEAR)
    whiteish = sat <= int(white_s_max)

    # Footprint and range geometry are constant for a fixed camera, so they
    # are computed once and cached (measured 58.5 ms at fine resolution, still
    # 19.8 ms on the coarse grid -- per frame, per camera, for an answer that
    # never changes).

    # The warp leaves black outside the camera's ground footprint, and that
    # artificial edge is a huge step for any top-hat -- it produced a line of
    # detections tracing the footprint boundary out to y = -9.9 m. Pull the
    # valid region in so the operator never straddles that edge.
    line_px_for_erode = max(1, int(round(line_width_m / mpp)))
    erode_k = max(3, 2 * line_px_for_erode + 5) | 1
    known_coarse, valid = _geometry(H, img_bgr.shape, grid, fine, k,
                                    max_range_m, erode_k)

    shape = val.shape
    lines_fine = np.zeros(shape, bool)
    potholes_fine = np.zeros(shape, bool)
    line_thr = pothole_thr = 0.0
    n_lines = n_potholes = 0

    # ---- lines: top-hat sized to the tape width ------------------------
    if want_lines:
        line_px = max(1, int(round(line_width_m / mpp)))
        # Element must be WIDER than the line, or the opening keeps the line
        # and the top-hat returns nothing. Wider still also admits blur and
        # a little calibration error.
        # ELLIPSE here, RECT for the pothole element below. The shape only
        # costs 1.7 ms at this size (2.5 vs 0.8 on an 800x800 BEV) and it
        # matters: a rectangular element leaves a straight, elongated
        # artefact along the edge of a large bright slab, which then passes
        # the aspect gate and reads as a line. At the pothole's 39x39 the
        # same choice costs 41.9 ms instead of 2.7, and there the circularity
        # and diameter gates make the element's shape irrelevant.
        se = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * line_px + 3, 2 * line_px + 3))
        resp = cv2.morphologyEx(val, cv2.MORPH_TOPHAT, se)
        line_thr = _robust_threshold(resp, valid, response_floor, mad_k)
        cand = (resp >= line_thr) & valid & whiteish
        for c in _components(cand.astype(np.uint8) * 255):
            (_, (rw, rh), angle) = cv2.minAreaRect(c)
            long_m, short_m = max(rw, rh) * mpp, min(rw, rh) * mpp
            if long_m < min_line_length_m or short_m > max_line_width_m:
                continue
            # Metric elongation. This is the same idea as the old image-space
            # min_elong, but here it means what it says: in BEV the ratio is
            # the line's true shape on the ground, not an artefact of how far
            # away it happened to be. It rejects the CORNER of a large bright
            # object, which a top-hat cannot suppress (the element will not
            # fit into a convex corner) and which reads as dark on both
            # perpendicular samples. Measured: a slab corner came through as
            # 0.42 x 0.25 m, aspect 1.7.
            if long_m < min_aspect * max(short_m, 1e-6):
                continue
            # minAreaRect's angle names the WIDTH side; if height is the long
            # one the long axis is 90 degrees round from it.
            long_axis_deg = angle if rw >= rh else angle + 90.0
            comp = _fill(shape, c)
            if _ridge_contrast(val, comp, line_px, long_axis_deg,
                               valid) < min_ridge_contrast:
                continue
            lines_fine |= comp
            n_lines += 1

    # ---- potholes: same operator, disc-sized element --------------------
    if want_potholes:
        disc_px = max(3, int(round(pothole_diameter_m / mpp)))
        # Element must exceed the disc, or the opening keeps the disc and the
        # top-hat returns nothing -- same rule as the line element above.
        se_px = int(disc_px * 1.6) | 1
        se_p = cv2.getStructuringElement(cv2.MORPH_RECT, (se_px, se_px))
        resp_p = cv2.morphologyEx(val, cv2.MORPH_TOPHAT, se_p)
        pothole_thr = _robust_threshold(resp_p, valid, response_floor,
                                        mad_k)
        cand_p = (resp_p >= pothole_thr) & valid & whiteish
        cand_p = cv2.morphologyEx(cand_p.astype(np.uint8) * 255,
                                  cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        lo = pothole_diameter_m * (1.0 - pothole_diameter_tol)
        hi = pothole_diameter_m * (1.0 + pothole_diameter_tol)
        for c in _components(cand_p):
            area = cv2.contourArea(c)
            if area <= 0:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < min_circularity:
                continue
            diam_m = 2.0 * np.sqrt(area / np.pi) * mpp
            if not (lo <= diam_m <= hi):
                continue
            potholes_fine |= _fill(shape, c)
            n_potholes += 1

    return {
        "lines": _maxpool_bool(lines_fine, k),
        "potholes": _maxpool_bool(potholes_fine, k),
        "known": known_coarse,
        "lines_fine": lines_fine,
        "potholes_fine": potholes_fine,
        "line_thr": line_thr,
        "pothole_thr": pothole_thr,
        "n_lines": n_lines,
        "n_potholes": n_potholes,
        "mpp": mpp,
    }
