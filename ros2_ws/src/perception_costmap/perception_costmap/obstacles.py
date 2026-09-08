"""
obstacles.py
------------
Obstacle detection from two sources:

- Camera (image space): contrast-based blobs that sit on the road but don't
  match its colour, or YOLO vehicle boxes. Factored from Adam Castillo's
  ``perception/costmap.py``. Returns an image-space boolean mask.

- Lidar (metric): drop the ground plane, keep returns within a height band,
  bin the survivors into the costmap grid. The most reliable obstacle source
  because lidar points are already metric. Returns both the filtered points
  (for republishing as a PointCloud2) and a grid-space obstacle mask.
"""

import numpy as np
import cv2

from .occupancy import GridSpec
from . import bev


# --------------------------------------------------------------------------
# Camera obstacles
# --------------------------------------------------------------------------
def detect_obstacles_camera(img_bgr, road_mask, min_area: int = 150) -> np.ndarray:
    """
    Blobs that lie within the road's extent but aren't road-coloured (cars,
    cones). Uses the convex hull of the road so cars -- which punch holes in
    the road mask -- are still counted as "on the road". Returns a boolean
    mask the size of the image.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    not_road_color = (s > 60) | (v < 40) | (v > 200)

    road_u8 = (road_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(road_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extent = np.zeros_like(road_u8)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        cv2.drawContours(extent, [hull], -1, 255, thickness=cv2.FILLED)

    raw = (not_road_color & (extent > 0)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    raw = cv2.dilate(raw, kernel, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    clean = np.zeros_like(raw, dtype=bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean[labels == i] = True
    return clean


def boxes_to_footprint_mask(boxes_xyxy, image_shape, footprint_frac=0.25):
    """
    Rasterise detector boxes as *ground-contact strips*, not full boxes.

    IPM assumes every pixel lies on the ground plane. A vehicle's upper pixels
    are 1-2 m above it, so warping a full box smears "obstacle" many metres
    down-range. Only the bottom footprint_frac of each box (where the object
    meets the road) is geometrically valid to project.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        strip = max(1, int(round((y2 - y1) * footprint_frac)))
        r0, r1 = max(0, y2 - strip), min(h, y2)
        c0, c1 = max(0, x1), min(w, x2)
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask


class YoloObstacleDetector:
    """YOLOv8 wrapper that loads the network ONCE (the old per-call load was
    ~100x slower than inference itself). Pass a .engine file to run TensorRT
    on the Jetson -- ultralytics handles both formats."""

    DEFAULT_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")

    def __init__(self, weights="yolov8n.pt", classes=DEFAULT_CLASSES,
                 conf=0.35, footprint_frac=0.25, device=None):
        from ultralytics import YOLO          # lazy: optional dependency
        self.model = YOLO(weights, task="detect")
        self.classes = set(classes)
        self.conf = conf
        self.footprint_frac = footprint_frac
        self.device = device

    # COCO label -> danger class. A pedestrian needs a much wider berth than a
    # parked car (see occupancy.DEFAULT_OBSTACLE_CLASSES), so the detector's
    # class must survive into the costmap instead of being flattened into one
    # anonymous "obstacle" blob.
    CLASS_GROUPS = {
        "person": "person",
        "bicycle": "person",      # a rider is a person; treat as vulnerable
        "motorcycle": "person",
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
    }

    def detect(self, img_bgr):
        """Union mask of every detected class (back-compat)."""
        grouped = self.detect_grouped(img_bgr)
        out = None
        for m in grouped.values():
            out = m if out is None else (out | m)
        if out is None:
            import numpy as _np
            out = _np.zeros(img_bgr.shape[:2], bool)
        return out

    def detect_grouped(self, img_bgr):
        """{group_name: footprint mask} -- 'person' and 'vehicle' kept apart."""
        res = self.model(img_bgr, verbose=False, conf=self.conf,
                         device=self.device)[0]
        per_group = {}
        for b in res.boxes:
            name = res.names[int(b.cls[0])]
            if name not in self.classes:
                continue
            group = self.CLASS_GROUPS.get(name, "generic")
            per_group.setdefault(group, []).append(b.xyxy[0].tolist())
        return {g: boxes_to_footprint_mask(boxes, img_bgr.shape,
                                           self.footprint_frac)
                for g, boxes in per_group.items()}


def cone_box_to_mask(img_bgr, box):
    """Area mask for one detected cone: orange + white-band gate inside the
    box, convex-hulled; falls back to the box's lower 85% when the gate
    finds nothing (dark / odd-colored cone). Boxes are never used directly.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (2, 90, 70), (22, 255, 255))       # orange body
    m |= cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))      # white bands
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    out = np.zeros((h, w), bool)
    if m.mean() > 0.10 * 255:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(np.vstack([c.reshape(-1, 2) for c in cnts]))
        mm = np.zeros(m.shape, np.uint8)
        cv2.fillPoly(mm, [hull], 255)
        out[y1:y2, x1:x2] = mm > 0
    else:
        out[y1 + int(0.15 * (y2 - y1)):y2, x1:x2] = True
    return out


class ConeDetector:
    """Dedicated cone detector (driving_seg/models/cone_det.pt --
    ExStella/Traffic-cones, Apache-2.0; .engine on the Jetson). Detections
    become AREA masks via cone_box_to_mask; catches striped/white-banded
    cones that color gates and COCO models miss. Loads once, like
    YoloObstacleDetector."""

    def __init__(self, weights="cone_det.pt", conf=0.35, device=None):
        from ultralytics import YOLO          # lazy: optional dependency
        self.model = YOLO(weights, task="detect")
        self.conf = conf
        self.device = device

    def detect(self, img_bgr):
        res = self.model(img_bgr, verbose=False, conf=self.conf,
                         device=self.device)[0]
        mask = np.zeros(img_bgr.shape[:2], bool)
        for b in res.boxes:
            m = cone_box_to_mask(img_bgr, b.xyxy[0].tolist())
            if m is not None:
                mask |= m
        return mask


def detect_obstacles_yolo(img_bgr, classes=YoloObstacleDetector.DEFAULT_CLASSES):
    """One-shot convenience kept for scripts. For anything per-frame use
    YoloObstacleDetector so the model loads once."""
    return YoloObstacleDetector(classes=classes).detect(img_bgr)


# --------------------------------------------------------------------------
# Lidar obstacles
# --------------------------------------------------------------------------
def filter_obstacle_points(points_xyz: np.ndarray,
                           z_min: float = 0.2,
                           z_max: float = 2.5) -> np.ndarray:
    """
    Keep points whose height is above the ground band and below the roofline.
    points_xyz: (N,3) in the robot frame (x fwd, y left, z up). Assumes roughly
    flat ground near the car. Returns the surviving (M,3) points.
    """
    if points_xyz.size == 0:
        return points_xyz.reshape(0, 3)
    z = points_xyz[:, 2]
    return points_xyz[(z >= z_min) & (z <= z_max)]


def points_to_grid_mask(points_xyz: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Bin metric obstacle points into a grid-space boolean obstacle mask.
    Vectorized: floor to cell indices, keep in-bounds, scatter.
    Uses floor (not int truncation), so points just below x_min/y_min are dropped rather than binned into border cells."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    if points_xyz.size == 0:
        return mask
    cols = np.floor((points_xyz[:, 0] - grid.x_min) / grid.resolution).astype(np.int64)
    rows = np.floor((points_xyz[:, 1] - grid.y_min) / grid.resolution).astype(np.int64)
    ok = (cols >= 0) & (cols < grid.width) & (rows >= 0) & (rows < grid.height)
    mask[rows[ok], cols[ok]] = True
    return mask


def depth_mask_to_grid(image_mask, depth_m, K, cam_xyz, pitch_deg, yaw_deg,
                       grid, min_depth=0.3, max_depth=20.0,
                       min_points=8, dilation_m=0.15, confidence=None,
                       max_confidence=70.0, edge_erode_px=1,
                       depth_mad_scale=3.5, depth_relative_tolerance=0.12,
                       return_stats=False):
    """Project masked registered-depth pixels directly into the metric grid.

    Returns ``None`` when depth is unavailable or insufficient so callers can
    explicitly fall back to ground-plane IPM. Otherwise returns a grid mask.
    """
    stats = {
        "mask_points": int(np.count_nonzero(image_mask)),
        "valid_points": 0,
        "confidence_rejected": 0,
        "outlier_rejected": 0,
        "confidence_available": confidence is not None,
    }

    def finished(result):
        return (result, stats) if return_stats else result

    if depth_m is None or K is None or image_mask.shape != depth_m.shape:
        return finished(None)
    if confidence is not None and confidence.shape != depth_m.shape:
        confidence = None
        stats["confidence_available"] = False

    source_mask = image_mask.astype(np.uint8)
    if edge_erode_px > 0:
        size = 2 * int(edge_erode_px) + 1
        eroded = cv2.erode(source_mask, np.ones((size, size), np.uint8))
        if np.count_nonzero(eroded) >= int(min_points):
            source_mask = eroded

    valid_range = (np.isfinite(depth_m)
                   & (depth_m >= min_depth) & (depth_m <= max_depth))
    if confidence is not None:
        confidence_ok = np.isfinite(confidence) & (confidence <= max_confidence)
        stats["confidence_rejected"] = int(np.count_nonzero(
            source_mask.astype(bool) & valid_range & ~confidence_ok))
        valid_range &= confidence_ok

    # Robustly reject background bleed at detection boundaries. Work per
    # connected component so nearby objects at different ranges survive.
    _, labels = cv2.connectedComponents(source_mask)
    accepted = np.zeros_like(source_mask, dtype=bool)
    for label in range(1, int(labels.max()) + 1):
        component = (labels == label) & valid_range
        values = depth_m[component]
        if values.size < int(min_points):
            continue
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        tolerance = max(
            0.08,
            float(depth_relative_tolerance) * median,
            float(depth_mad_scale) * 1.4826 * mad,
        )
        keep = component & (np.abs(depth_m - median) <= tolerance)
        stats["outlier_rejected"] += int(np.count_nonzero(component & ~keep))
        accepted |= keep

    rows, cols = np.nonzero(accepted)
    stats["valid_points"] = int(len(rows))
    if len(rows) < int(min_points):
        return finished(None)

    # Bound projection cost for large masks without biasing one image region.
    stride = max(1, int(np.ceil(len(rows) / 3000.0)))
    rows, cols = rows[::stride], cols[::stride]
    z = depth_m[rows, cols].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    optical = np.column_stack(((cols - cx) * z / fx,
                               (rows - cy) * z / fy,
                               z))
    robot = bev.optical_points_to_robot(
        optical, cam_xyz, pitch_deg, yaw_deg)
    result = points_to_grid_mask(robot, grid)
    radius = max(0, int(round(float(dilation_m) / grid.resolution)))
    if radius and result.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        result = cv2.dilate(result.astype(np.uint8), kernel).astype(bool)
    return finished(result)
