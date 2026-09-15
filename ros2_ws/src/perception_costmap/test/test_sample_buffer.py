import math

import numpy as np

from perception_costmap.sample_buffer import PoseBuffer, TimestampedBuffer


def test_nearest_sample_respects_tolerance():
    buffer = TimestampedBuffer(maxlen=3)
    buffer.add(1.00, "old")
    buffer.add(1.10, "new")
    assert buffer.nearest(1.08, 0.05) == (1.10, "new")
    assert buffer.nearest(1.30, 0.05) is None


def test_buffer_drops_oldest_sample():
    buffer = TimestampedBuffer(maxlen=2)
    buffer.add(1.0, "one")
    buffer.add(2.0, "two")
    buffer.add(3.0, "three")
    assert [value for _, value in buffer.samples] == ["two", "three"]


def test_pose_interpolation_uses_capture_timestamp():
    poses = PoseBuffer()
    poses.add(10.0, (0.0, 0.0, 0.0))
    poses.add(10.1, (1.0, 2.0, 0.2))
    assert np.allclose(poses.interpolate(10.05, 0.1), (0.5, 1.0, 0.1))


def test_pose_interpolation_wraps_yaw_on_short_arc():
    poses = PoseBuffer()
    poses.add(1.0, (0.0, 0.0, math.radians(179.0)))
    poses.add(2.0, (0.0, 0.0, math.radians(-179.0)))
    _, _, yaw = poses.interpolate(1.5, 1.0)
    assert abs(abs(yaw) - math.pi) < 1e-6


def test_pose_interpolation_rejects_large_time_gap():
    poses = PoseBuffer()
    poses.add(1.0, (0.0, 0.0, 0.0))
    poses.add(2.0, (1.0, 0.0, 0.0))
    assert poses.interpolate(1.5, 0.2) is None
