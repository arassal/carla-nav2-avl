"""Offline tests for segmentation + obstacle detection (no ROS required)."""

import numpy as np
import cv2

from perception_costmap.segmentation import segment_road
from perception_costmap import obstacles
from perception_costmap.occupancy import GridSpec


def _demo_image(w=640, h=480):
    img = np.full((h, w, 3), (60, 110, 60), np.uint8)        # green grass
    cv2.rectangle(img, (0, h // 3), (w, 2 * h // 3), (90, 90, 90), -1)  # gray road
    cv2.rectangle(img, (250, h // 3 + 5), (320, 2 * h // 3 - 35), (30, 30, 200), -1)  # red car
    return img


def test_road_segmentation_finds_the_band():
    img = _demo_image()
    road = segment_road(img)
    h = img.shape[0]
    # middle band (road) should be mostly road; top (grass) should not
    assert road[h // 2].mean() > 0.5
    assert road[h // 10].mean() < 0.1


def test_camera_obstacle_detected_on_road():
    img = _demo_image()
    road = segment_road(img)
    obst = obstacles.detect_obstacles_camera(img, road)
    assert obst.any()                       # the red car is found
    # it sits in the road band, not in the sky
    ys = np.where(obst.any(axis=1))[0]
    assert ys.min() > img.shape[0] // 4


def test_lidar_ground_filter_and_binning():
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    pts = np.array([
        [5.0, 0.0, 0.0],    # ground -> dropped
        [5.0, 0.0, 1.0],    # obstacle -> kept
        [3.0, 2.0, 0.8],    # obstacle -> kept
        [5.0, 0.0, 5.0],    # above roofline -> dropped
    ])
    kept = obstacles.filter_obstacle_points(pts, z_min=0.2, z_max=2.5)
    assert len(kept) == 2
    mask = obstacles.points_to_grid_mask(kept, g)
    assert mask.sum() == 2
    col, row = g.world_to_cell(5.0, 0.0)
    assert mask[row, col]


def test_empty_lidar_is_safe():
    g = GridSpec(resolution=0.5)
    kept = obstacles.filter_obstacle_points(np.zeros((0, 3)))
    assert kept.shape == (0, 3)
    assert not obstacles.points_to_grid_mask(kept, g).any()


def test_depth_mask_projects_obstacle_to_metric_grid():
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    depth = np.full((100, 100), np.nan, np.float32)
    image_mask = np.zeros((100, 100), bool)
    image_mask[45:55, 45:55] = True
    depth[image_mask] = 5.0
    K = np.array([[100.0, 0.0, 50.0],
                  [0.0, 100.0, 50.0],
                  [0.0, 0.0, 1.0]])
    projected = obstacles.depth_mask_to_grid(
        image_mask, depth, K, (0.0, 0.0, 1.0), 0.0, 0.0, g,
        dilation_m=0.0)
    assert projected is not None and projected.any()
    cols = np.where(projected)[1]
    x = g.x_min + (cols.mean() + 0.5) * g.resolution
    assert abs(x - 5.0) < 0.2


def test_depth_projection_requests_fallback_when_depth_is_invalid():
    g = GridSpec()
    mask = np.ones((10, 10), bool)
    depth = np.full((10, 10), np.nan, np.float32)
    assert obstacles.depth_mask_to_grid(
        mask, depth, np.eye(3), (0, 0, 1), 0, 0, g) is None


def test_depth_projection_rejects_low_confidence_pixels():
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    mask = np.zeros((40, 40), bool)
    mask[10:30, 10:30] = True
    depth = np.full((40, 40), 4.0, np.float32)
    confidence = np.zeros((40, 40), np.float32)
    confidence[10:30, 10:20] = 90.0
    projected, stats = obstacles.depth_mask_to_grid(
        mask, depth, np.array([[100.0, 0, 20], [0, 100.0, 20], [0, 0, 1]]),
        (0, 0, 1), 0, 0, g, confidence=confidence,
        max_confidence=70, dilation_m=0, return_stats=True)
    assert projected is not None and projected.any()
    assert stats["confidence_available"]
    assert stats["confidence_rejected"] > 0


def test_depth_projection_rejects_background_depth_bleed():
    g = GridSpec(x_min=0, x_max=15, y_min=-5, y_max=5, resolution=0.1)
    mask = np.zeros((50, 50), bool)
    mask[10:40, 10:40] = True
    depth = np.full((50, 50), np.nan, np.float32)
    depth[mask] = 4.0
    depth[10:40, 36:40] = 10.0
    projected, stats = obstacles.depth_mask_to_grid(
        mask, depth, np.array([[100.0, 0, 25], [0, 100.0, 25], [0, 0, 1]]),
        (0, 0, 1), 0, 0, g, dilation_m=0, return_stats=True)
    assert projected is not None
    occupied_x = np.where(projected)[1] * g.resolution + g.x_min
    assert occupied_x.max() < 6.0
    assert stats["outlier_rejected"] > 0


def test_depth_projection_keeps_disconnected_objects_at_different_ranges():
    g = GridSpec(x_min=0, x_max=12, y_min=-5, y_max=5, resolution=0.1)
    mask = np.zeros((60, 80), bool)
    mask[20:40, 10:30] = True
    mask[20:40, 50:70] = True
    depth = np.full(mask.shape, np.nan, np.float32)
    depth[20:40, 10:30] = 3.0
    depth[20:40, 50:70] = 8.0
    projected = obstacles.depth_mask_to_grid(
        mask, depth, np.array([[100.0, 0, 40], [0, 100.0, 30], [0, 0, 1]]),
        (0, 0, 1), 0, 0, g, dilation_m=0)
    occupied_x = np.where(projected)[1] * g.resolution + g.x_min
    assert occupied_x.min() < 4.0
    assert occupied_x.max() > 7.0


def test_depth_projection_reports_when_confidence_rejects_everything():
    g = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.1)
    mask = np.ones((20, 20), bool)
    depth = np.full(mask.shape, 4.0, np.float32)
    confidence = np.full(mask.shape, 90.0, np.float32)
    projected, stats = obstacles.depth_mask_to_grid(
        mask, depth, np.array([[100.0, 0, 10], [0, 100.0, 10], [0, 0, 1]]),
        (0, 0, 1), 0, 0, g, confidence=confidence,
        max_confidence=70, return_stats=True)
    assert projected is None
    assert stats["confidence_rejected"] > 0
