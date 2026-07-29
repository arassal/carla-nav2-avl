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
