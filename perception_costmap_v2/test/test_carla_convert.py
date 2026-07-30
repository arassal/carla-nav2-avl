import numpy as np
from perception_costmap.carla_convert import (
    bgra_bytes_to_bgr, carla_lidar_to_rep103, semantic_to_road_mask, mask_iou,
)


def test_bgra_bytes_to_bgr_shape_and_channel_drop():
    h, w = 4, 5
    raw = np.arange(h * w * 4, dtype=np.uint8).tobytes()
    bgr = bgra_bytes_to_bgr(raw, h, w)
    assert bgr.shape == (h, w, 3)


def test_carla_lidar_to_rep103_flips_y_and_adds_height():
    # x, y_right, z, intensity
    raw = np.array([[1.0, 2.0, 0.5, 1.0],
                    [3.0, -4.0, 0.1, 1.0]], dtype=np.float32).tobytes()
    pts = carla_lidar_to_rep103(raw, sensor_z=1.8)
    assert pts.shape == (2, 3)
    assert pts[0, 0] == 1.0
    assert pts[0, 1] == -2.0          # y flipped for left-handed -> REP-103
    assert pts[0, 2] == 0.5 + 1.8     # mount height added
    assert pts[1, 1] == 4.0


def test_semantic_to_road_mask_matches_tags():
    sem = np.zeros((4, 4, 3), dtype=np.uint8)
    sem[0, 0, 2] = 1     # road tag, in red channel (index 2 of BGR)
    sem[1, 1, 2] = 24    # roadline tag
    sem[2, 2, 2] = 7     # unrelated tag (e.g. traffic sign)
    mask = semantic_to_road_mask(sem, road_tags=(1, 24))
    assert mask[0, 0] and mask[1, 1]
    assert not mask[2, 2]
    assert mask.sum() == 2


def test_mask_iou_identical_masks():
    a = np.ones((5, 5), dtype=bool)
    assert mask_iou(a, a) == 1.0


def test_mask_iou_disjoint_masks():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0, 0] = True
    b[3, 3] = True
    assert mask_iou(a, b) == 0.0


def test_mask_iou_both_empty_is_one():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    assert mask_iou(a, b) == 1.0


def test_mask_iou_partial_overlap():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[0:2, 0:2] = True   # 4 cells
    b[1:3, 1:3] = True   # 4 cells, overlap at (1,1) = 1 cell
    # union = 4+4-1 = 7, intersection = 1
    assert abs(mask_iou(a, b) - (1 / 7)) < 1e-9
