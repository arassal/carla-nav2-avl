import numpy as np
from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103, semantic_to_road_mask, mask_iou)


def test_bgra_to_bgr():
    raw = np.array([[10, 20, 30, 255]], np.uint8).tobytes()   # 1x1 BGRA
    img = bgra_bytes_to_bgr(raw, 1, 1)
    assert img.shape == (1, 1, 3)
    assert list(img[0, 0]) == [10, 20, 30]


def test_lidar_flips_y_and_offsets_z():
    # CARLA lidar: (x, y_right, z, intensity) float32
    raw = np.array([[5.0, 2.0, 0.5, 0.9]], np.float32).tobytes()
    pts = carla_lidar_to_rep103(raw, sensor_z=1.8)
    assert pts.shape == (1, 3)
    assert pts[0, 0] == 5.0
    assert pts[0, 1] == -2.0          # left-handed -> REP-103
    assert abs(pts[0, 2] - 2.3) < 1e-6   # sensor frame -> base_link height


def test_semantic_road_mask_reads_red_channel():
    sem = np.zeros((2, 2, 3), np.uint8)
    sem[0, 0, 2] = 1      # Roads tag in R channel
    sem[1, 1, 2] = 24     # RoadLines
    m = semantic_to_road_mask(sem, road_tags=(1, 24))
    assert m[0, 0] and m[1, 1] and m.sum() == 2


def test_mask_iou():
    a = np.array([[True, True], [False, False]])
    b = np.array([[True, False], [False, False]])
    assert abs(mask_iou(a, b) - 0.5) < 1e-9
    assert mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 1.0
