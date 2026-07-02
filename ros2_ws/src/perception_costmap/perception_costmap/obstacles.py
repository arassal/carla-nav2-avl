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


def detect_obstacles_yolo(img_bgr, classes=("car", "truck", "bus", "person")) -> np.ndarray:
    """YOLO vehicle/person boxes rasterised to an image-space mask. Optional
    dependency: ``pip install ultralytics``."""
    from ultralytics import YOLO          # lazy: optional dependency
    model = YOLO("yolov8n.pt")
    res = model(img_bgr, verbose=False)[0]
    mask = np.zeros(img_bgr.shape[:2], dtype=bool)
    for box in res.boxes:
        if res.names[int(box.cls[0])] in classes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask[y1:y2, x1:x2] = True
    return mask


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
