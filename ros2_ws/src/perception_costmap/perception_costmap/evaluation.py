"""ROS-free perception accuracy metrics."""

import numpy as np


def binary_metrics(prediction, truth):
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth shapes differ")
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    union = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (2.0 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "iou": tp / union if union else 1.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def projection_errors(H, image_points, ground_points, grid):
    """Return metric XY errors for image-to-ground calibration samples."""
    image_points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    ground_points = np.asarray(ground_points, dtype=float).reshape(-1, 2)
    if len(image_points) != len(ground_points) or not len(image_points):
        raise ValueError("equal nonempty image and ground point sets required")
    homogeneous = np.column_stack((image_points, np.ones(len(image_points))))
    projected = (np.asarray(H, dtype=float) @ homogeneous.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-9):
        raise ValueError("point projects to infinity")
    cells = projected[:, :2] / projected[:, 2:3]
    predicted = np.column_stack((
        grid.x_min + cells[:, 0] * grid.resolution,
        grid.y_min + cells[:, 1] * grid.resolution))
    return np.linalg.norm(predicted - ground_points, axis=1)
