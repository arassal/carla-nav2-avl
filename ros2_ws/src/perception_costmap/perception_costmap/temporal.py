"""
temporal.py — per-cell obstacle confidence over time.

A costmap built from single frames flickers: one noisy frame paints a lethal
cell and the planner reacts. This accumulator requires evidence to mark AND
evidence to clear:

  hit:   cell detected as obstacle this frame        conf += hit
  miss:  cell observed (in FOV / lidar) and empty    conf -= miss
  else:  not observed                                conf unchanged

Report lethal where conf >= threshold. Pure numpy, ROS-free.
"""

import numpy as np
import cv2


def reproject_grid(array, previous_pose, current_pose, grid,
                   interpolation=cv2.INTER_NEAREST):
    """Reproject a robot-frame grid between two odometry poses."""
    px, py, pa = (float(v) for v in previous_pose)
    cx, cy, ca = (float(v) for v in current_pose)
    rows, cols = np.indices(array.shape, dtype=np.float32)
    x_cur = grid.x_min + (cols + 0.5) * grid.resolution
    y_cur = grid.y_min + (rows + 0.5) * grid.resolution

    cos_c, sin_c = np.cos(ca), np.sin(ca)
    x_world = cx + cos_c * x_cur - sin_c * y_cur
    y_world = cy + sin_c * x_cur + cos_c * y_cur
    dx, dy = x_world - px, y_world - py
    cos_p, sin_p = np.cos(pa), np.sin(pa)
    x_prev = cos_p * dx + sin_p * dy
    y_prev = -sin_p * dx + cos_p * dy
    map_x = ((x_prev - grid.x_min) / grid.resolution - 0.5).astype(np.float32)
    map_y = ((y_prev - grid.y_min) / grid.resolution - 0.5).astype(np.float32)
    return cv2.remap(
        array, map_x, map_y, interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class TemporalObstacleFilter:
    def __init__(self, shape, hit=0.4, miss=0.2, threshold=0.5):
        self.hit = float(hit)
        self.miss = float(miss)
        self.threshold = float(threshold)
        self.conf = np.zeros(shape, dtype=np.float32)

    def update(self, obstacle_mask, observed_mask):
        obstacle_mask = obstacle_mask.astype(bool)
        # decay anywhere we looked and saw nothing — including cells that only
        # ever had lidar evidence (conf > 0) so stale marks can't live forever
        decay = (observed_mask.astype(bool) & ~obstacle_mask)
        self.conf[obstacle_mask] += self.hit
        self.conf[decay] -= self.miss
        np.clip(self.conf, 0.0, 1.0, out=self.conf)
        return self.conf >= self.threshold

    def compensate_motion(self, previous_pose, current_pose, grid):
        """Reproject confidence from the previous base frame to the current.

        Poses are ``(x, y, yaw)`` in a shared odometry frame. Reverse mapping
        is used so every destination cell samples the previous confidence map.
        """
        if previous_pose is None or current_pose is None:
            return
        self.conf = reproject_grid(
            self.conf, previous_pose, current_pose, grid,
            interpolation=cv2.INTER_LINEAR)
