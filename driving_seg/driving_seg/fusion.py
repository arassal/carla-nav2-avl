"""Compose per-class boolean masks into one highlighted frame.

Pure numpy/cv2 — no model deps, fully offline-testable. No bounding boxes,
ever: translucent area fills + a brightened contour per region.
"""

import cv2
import numpy as np

from .config import CLASSES, OVERLAY_ALPHA, CONTOUR_GAIN, paint_order


def _validate(masks, shape):
    out = {}
    for name, m in masks.items():
        if name not in CLASSES or m is None:
            continue
        m = np.asarray(m)
        if m.dtype != bool:
            m = m > 0
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        out[name] = m
    return out


def claim_map(masks, shape):
    """Resolve overlaps: each pixel belongs to its highest-priority class.
    Returns int map (0 = background, else CLASSES[name]['priority'])."""
    claim = np.zeros(shape, np.uint8)
    for name in paint_order():                    # low -> high: high overwrites
        if name in masks:
            claim[masks[name]] = CLASSES[name]["priority"]
    return claim


def compose(frame_bgr, masks, alpha=OVERLAY_ALPHA):
    """frame + {class: bool HxW} -> highlighted BGR frame."""
    h, w = frame_bgr.shape[:2]
    masks = _validate(masks, (h, w))
    claim = claim_map(masks, (h, w))

    out = frame_bgr.copy()
    color_plane = np.zeros_like(frame_bgr)
    painted = claim > 0
    for name in paint_order():
        if name not in masks:
            continue
        mine = claim == CLASSES[name]["priority"]
        color_plane[mine] = CLASSES[name]["color"]
    if painted.any():
        blend = cv2.addWeighted(frame_bgr, 1.0 - alpha, color_plane, alpha, 0)
        out[painted] = blend[painted]

    # brightened contour outlines the exclusive region of each class
    for name in paint_order():
        if name not in masks:
            continue
        mine = (claim == CLASSES[name]["priority"]).astype(np.uint8)
        if not mine.any():
            continue
        contours, _ = cv2.findContours(mine, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        col = tuple(min(255, int(c * CONTOUR_GAIN)) for c in CLASSES[name]["color"])
        cv2.drawContours(out, contours, -1, col, 2, cv2.LINE_AA)
    return out


def legend(frame_bgr, present):
    """Draw a small legend for the classes present in this frame."""
    y = 22
    for name in paint_order()[::-1]:
        if name not in present:
            continue
        col = CLASSES[name]["color"]
        cv2.rectangle(frame_bgr, (8, y - 11), (24, y + 3), col, -1)
        cv2.putText(frame_bgr, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
    return frame_bgr
