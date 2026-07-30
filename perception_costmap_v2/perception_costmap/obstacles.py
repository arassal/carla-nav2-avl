"""
obstacles.py -- obstacle evidence from lidar and camera.

Two independent sources feed the fused obstacle mask:
  1. lidar points (metric, ground truth range) -> points_to_grid_mask
  2. camera detections (classical contrast, or YOLO) -> a grid mask via
     boxes_to_footprint_mask

Key correctness point: warping a full detection box through IPM smears the
obstacle behind the vehicle, because IPM assumes every pixel lies on the
ground plane -- true only for the *bottom* of the box. So we rasterize only
a thin footprint strip at the box's bottom edge.

torch/ultralytics stay a lazy import: this module (and its tests) must run
with only numpy + opencv installed.
"""

from dataclasses import dataclass
import numpy as np
import cv2

from .occupancy import GridSpec
from .bev import warp_to_bev


def points_to_grid_mask(points_xyz: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Bin metric obstacle points (already in the robot/base_link frame,
    ground already removed by the caller) into a grid-space boolean mask.
    Vectorized: floor to cell indices, keep in-bounds, scatter."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    if points_xyz.size == 0:
        return mask
    cols = np.floor((points_xyz[:, 0] - grid.x_min) / grid.resolution).astype(np.int64)
    rows = np.floor((points_xyz[:, 1] - grid.y_min) / grid.resolution).astype(np.int64)
    ok = (cols >= 0) & (cols < grid.width) & (rows >= 0) & (rows < grid.height)
    mask[rows[ok], cols[ok]] = True
    return mask


def remove_ground_plane(points_xyz: np.ndarray, z_min: float = -0.3,
                        z_max: float = 3.0) -> np.ndarray:
    """Keep only points in a z-band above the ground and below a sane
    overhead cutoff. Assumes roughly flat ground near the vehicle -- a
    documented limitation, not a full plane fit."""
    if points_xyz.size == 0:
        return points_xyz
    z = points_xyz[:, 2]
    return points_xyz[(z >= z_min) & (z <= z_max)]


def boxes_to_footprint_mask(boxes_xyxy, image_shape, footprint_frac: float = 0.25) -> np.ndarray:
    """
    Pure rasterizer: given detection boxes (x1,y1,x2,y2 in pixels) and the
    source image shape, return a bool mask (same H,W as the image) that is
    True only in a thin strip at the BOTTOM of each box -- the part of the
    detection that plausibly touches the ground plane.

    ``footprint_frac`` is the fraction of the box height kept, measured from
    the bottom edge upward (0.25 = bottom quarter). This mask is what gets
    warped through IPM, not the full box -- warping the whole box would
    project points 1-2m up (a car's roof) onto ground metres behind the car.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for box in np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4):
        x1, y1, x2, y2 = box
        x1c, x2c = sorted((max(0, x1), min(w, x2)))
        y1c, y2c = sorted((max(0, y1), min(h, y2)))
        if x2c <= x1c or y2c <= y1c:
            continue
        box_h = y2c - y1c
        strip_top = y2c - box_h * footprint_frac
        r0 = int(np.floor(strip_top))
        r1 = int(np.ceil(y2c))
        c0 = int(np.floor(x1c))
        c1 = int(np.ceil(x2c))
        mask[max(0, r0):min(h, r1), max(0, c0):min(w, c1)] = True
    return mask


def detect_obstacles_classical(img_bgr, thresh: int = 40, min_area: int = 60) -> np.ndarray:
    """
    Cheap camera obstacle cue with no model weights: high local contrast
    against a smoothed background. Not accurate -- a fallback for when no
    detector is configured, and for tests that need "some obstacle mask"
    without ultralytics installed.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = cv2.absdiff(gray, blurred)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask, dtype=bool)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(out.view(np.uint8), [c], -1, 1, thickness=-1)
    return out


@dataclass
class Detection:
    box_xyxy: tuple
    cls_name: str
    conf: float


class YoloObstacleDetector:
    """
    Loads a YOLOv8 model ONCE (the #1 real-time bug in the original
    prototype was constructing `YOLO(...)` inside the per-frame detect
    call, reloading the network every tick). Accepts `.pt` or `.engine`
    weights transparently -- ultralytics loads TensorRT engines through the
    same API, which is the whole Jetson deployment path.

    `.detect()` returns a bool obstacle mask (footprint-strip rasterized,
    see boxes_to_footprint_mask) and `.detect_classed()` additionally
    returns per-class boxes so the caller can build per-class halos
    (occupancy.DEFAULT_OBSTACLE_CLASSES).
    """

    CLASS_MAP = {
        "person": "person",
        "car": "vehicle", "truck": "vehicle", "bus": "vehicle",
        "traffic cone": "cone", "cone": "cone",
    }

    def __init__(self, weights="yolov8n.pt", classes=None, conf=0.35,
                footprint_frac=0.25, device=None):
        self.weights = weights
        self.classes = classes
        self.conf = conf
        self.footprint_frac = footprint_frac
        self.device = device
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        except Exception:
            # Warm-load failure must not take the node down -- caller falls
            # back to detect_obstacles_classical.
            self._load_failed = True

    def detect_classed(self, img_bgr):
        """Returns list[Detection]. Empty list (not an exception) if the
        model failed to load or found nothing."""
        self._ensure_loaded()
        if self._model is None:
            return []
        results = self._model.predict(img_bgr, conf=self.conf,
                                      device=self.device, verbose=False)
        out = []
        for r in results:
            names = r.names
            for b in r.boxes:
                cls_id = int(b.cls[0])
                raw_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                mapped = self.CLASS_MAP.get(raw_name, "generic")
                if self.classes and mapped not in self.classes:
                    continue
                xyxy = tuple(float(v) for v in b.xyxy[0].tolist())
                out.append(Detection(xyxy, mapped, float(b.conf[0])))
        return out

    def detect(self, img_bgr) -> np.ndarray:
        """Legacy single-mask interface: all detections unioned into one
        footprint mask, regardless of class."""
        dets = self.detect_classed(img_bgr)
        if not dets:
            return np.zeros(img_bgr.shape[:2], dtype=bool)
        boxes = [d.box_xyxy for d in dets]
        return boxes_to_footprint_mask(boxes, img_bgr.shape, self.footprint_frac)

    def detect_per_class(self, img_bgr) -> dict:
        """Returns {class_name: bool mask} for building per-class inflation
        layers (see occupancy.build_cost_array(obstacle_layers=...))."""
        dets = self.detect_classed(img_bgr)
        by_class = {}
        for d in dets:
            by_class.setdefault(d.cls_name, []).append(d.box_xyxy)
        return {
            name: boxes_to_footprint_mask(boxes, img_bgr.shape, self.footprint_frac)
            for name, boxes in by_class.items()
        }


def camera_obstacle_mask_to_grid(mask_img_space: np.ndarray, H, grid: GridSpec,
                                 known_mask: np.ndarray = None) -> np.ndarray:
    """Warp an image-space obstacle mask into the BEV grid via the camera's
    IPM homography. Thin wrapper kept here so obstacles.py has one call for
    'camera obstacle -> grid mask', mirroring points_to_grid_mask for lidar.

    ``known_mask`` (pass the camera's ``bev_known_mask``) clips the warped
    result to cells the camera actually observes. This matters specifically
    for camera-derived masks: cv2.warpPerspective does inverse (nearest)
    sampling per destination cell, and a destination cell whose inverse-
    mapped source ray has near-zero or negative projective depth (i.e. it
    is behind the camera or beyond the horizon) can still land on real
    image content and get sampled -- the same "mirror cell" failure mode
    ``bev.bev_known_mask`` exists to detect. Left un-clipped, that can plant
    an isolated spurious obstacle cell outside the camera's real footprint.

    Lidar obstacles (points_to_grid_mask) don't have this failure mode --
    they're metric points, not warped image content -- and per
    occupancy.build_cost_array's own convention are deliberately never
    clipped (a spurious real-sensor obstacle is the safe direction)."""
    warped = warp_to_bev(mask_img_space.astype(np.uint8), H, grid,
                        interp=cv2.INTER_NEAREST)
    mask = warped.astype(bool)
    if known_mask is not None:
        mask &= known_mask
    return mask
