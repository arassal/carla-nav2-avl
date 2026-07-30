"""
YoloObstacleDetector must never crash the node just because ultralytics
isn't installed or the weights path is bogus -- it degrades to "detected
nothing" and the caller (costmap_node) is expected to fall back to the
classical detector. These tests run whether or not ultralytics is present.
"""

import numpy as np
from perception_costmap.obstacles import YoloObstacleDetector


def test_yolo_detector_bad_weights_does_not_raise():
    det = YoloObstacleDetector(weights="/definitely/not/a/real/path.pt")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = det.detect(img)
    assert mask.shape == (100, 100)
    assert not mask.any()


def test_yolo_detector_load_failure_is_cached():
    det = YoloObstacleDetector(weights="/definitely/not/a/real/path.pt")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    det.detect(img)
    assert det._load_failed is True
    # second call must not retry the (expensive) import/load
    det.detect(img)
    assert det._model is None


def test_yolo_detector_per_class_empty_on_failure():
    det = YoloObstacleDetector(weights="/definitely/not/a/real/path.pt")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    out = det.detect_per_class(img)
    assert out == {}
