"""
carla_convert.py — pure converters between CARLA sensor buffers and our
REP-103 / numpy conventions. No carla or ROS imports: these are the exact
functions where sim-to-real geometry bugs hide, so they are unit-tested.

Gotchas encoded here:
- CARLA uses a LEFT-handed frame (x fwd, y RIGHT, z up). REP-103 is y LEFT.
- The semantic camera writes the class tag into the red channel.
- Semantic tag ids changed in CARLA 0.9.13; verify with the printout in
  tools/carla_feed.py rather than trusting defaults blindly.
"""

import numpy as np


def bgra_bytes_to_bgr(raw, height, width):
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    return arr[:, :, :3].copy()


def carla_lidar_to_rep103(raw, sensor_z):
    """CARLA lidar buffer (x, y_right, z, intensity) -> (N,3) base_link
    points: flip y for handedness, add mount height so z is above ground."""
    pts = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[:, :3].copy()
    pts = pts.astype(np.float64)
    pts[:, 1] *= -1.0
    pts[:, 2] += float(sensor_z)
    return pts


def semantic_to_road_mask(sem_bgr, road_tags=(1, 24)):
    tags = sem_bgr[:, :, 2]
    mask = np.zeros(tags.shape, dtype=bool)
    for t in road_tags:
        mask |= (tags == t)
    return mask


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    if union == 0:
        return 1.0
    return float((a & b).sum()) / float(union)
