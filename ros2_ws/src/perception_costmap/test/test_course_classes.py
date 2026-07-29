"""Offline tests for the course classes (cones + white lines), no ROS/torch.

Asymmetric fixtures throughout (mirror-bug class: a feature on the LEFT
must land on the LEFT).
"""

import numpy as np
import cv2

from perception_costmap.obstacles import cone_box_to_mask
from perception_costmap.segmentation import white_line_mask


def _cone_image(w=320, h=240, cx=80):
    """Grey scene with an orange triangle (cone) centred at x=cx."""
    img = np.full((h, w, 3), 120, np.uint8)
    pts = np.array([[cx, 60], [cx - 30, 180], [cx + 30, 180]])
    cv2.fillPoly(img, [pts], (30, 110, 240))          # BGR orange
    return img


def test_cone_mask_covers_cone_and_stays_left():
    img = _cone_image(cx=80)
    m = cone_box_to_mask(img, (40, 50, 120, 190))
    assert m is not None and m.any()
    assert m[120, 80]                                  # cone body covered
    assert not m[:, 200:].any()                        # right half untouched
    ys, xs = np.nonzero(m)
    assert xs.mean() < img.shape[1] / 2                # mass on the LEFT


def test_cone_mask_box_fallback_when_colorless():
    img = np.full((240, 320, 3), 120, np.uint8)        # nothing orange
    m = cone_box_to_mask(img, (200, 100, 260, 200))
    assert m is not None and m.any()
    ys, xs = np.nonzero(m)
    assert xs.min() >= 200 and xs.max() < 260          # stays inside box
    assert ys.min() >= 100 + int(0.15 * 100)           # lower-box fallback


def test_cone_mask_degenerate_box():
    img = _cone_image()
    assert cone_box_to_mask(img, (10, 10, 11, 11)) is None


def test_white_line_on_grass_detected_left():
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = (40, 140, 40)                          # grass
    img[:, 60:75] = (245, 245, 245)                    # vertical line, LEFT
    m = white_line_mask(img)
    assert m[:, 60:75].mean() > 0.5                    # line found
    assert not m[:, 160:].any()                        # right half clean


def test_white_line_ignores_non_grass_scene():
    img = np.full((240, 320, 3), 120, np.uint8)        # indoor grey
    img[:, 60:75] = (245, 245, 245)
    assert not white_line_mask(img).any()


def test_white_line_rejects_round_blob():
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = (40, 140, 40)
    cv2.circle(img, (160, 120), 25, (245, 245, 245), -1)   # blob, not a line
    assert not white_line_mask(img).any()
