import numpy as np

from perception_costmap import bev
from perception_costmap.evaluation import binary_metrics, projection_errors
from perception_costmap.occupancy import GridSpec


def test_binary_metrics_counts_errors():
    truth = np.array([[1, 1], [0, 0]], bool)
    pred = np.array([[1, 0], [1, 0]], bool)
    metrics = binary_metrics(pred, truth)
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1
    assert abs(metrics["iou"] - 1 / 3) < 1e-9
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_projection_error_is_zero_for_calibration_points():
    grid = GridSpec(x_min=0, x_max=20, y_min=-10, y_max=10, resolution=0.1)
    image = [(200, 470), (440, 470), (380, 120), (260, 120)]
    ground = [(2, -3), (2, 3), (18, 3), (18, -3)]
    H = bev.homography_from_points(image, ground, grid)
    errors = projection_errors(H, image, ground, grid)
    assert errors.max() < 1e-4
