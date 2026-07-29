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
