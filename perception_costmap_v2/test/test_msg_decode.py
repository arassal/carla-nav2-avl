"""
Offline tests for util's Image/PointCloud2 decoders -- the subscription path
in costmap_node. These run with no ROS installed by feeding duck-typed
message objects, which is the whole reason the decoding lives in util.py
instead of behind cv_bridge.
"""

import numpy as np
import pytest

from perception_costmap.util import image_msg_to_bgr, pointcloud2_to_xyz


class FakeImage:
    def __init__(self, height, width, encoding, data, step=None, channels=3):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.step = step if step is not None else width * channels
        self.data = data


class FakeField:
    def __init__(self, name, offset, datatype=7):
        self.name = name
        self.offset = offset
        self.datatype = datatype
        self.count = 1


class FakeCloud:
    def __init__(self, fields, point_step, data, width, height=1, is_bigendian=False):
        self.fields = fields
        self.point_step = point_step
        self.data = data
        self.width = width
        self.height = height
        self.is_bigendian = is_bigendian


def test_bgr8_roundtrip():
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    msg = FakeImage(4, 5, "bgr8", img.tobytes())
    assert np.array_equal(image_msg_to_bgr(msg), img)


def test_rgb8_channels_are_swapped_not_just_copied():
    # asymmetric per-channel values: a decoder that ignored the encoding and
    # returned the buffer unchanged would still pass a symmetric fixture
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    out = image_msg_to_bgr(FakeImage(2, 2, "rgb8", rgb.tobytes()))
    assert (out[..., 0] == 30).all()   # B <- R
    assert (out[..., 1] == 20).all()
    assert (out[..., 2] == 10).all()   # R <- B


def test_bgra8_drops_alpha():
    bgra = np.zeros((3, 3, 4), dtype=np.uint8)
    bgra[..., :3] = [1, 2, 3]
    bgra[..., 3] = 255
    out = image_msg_to_bgr(FakeImage(3, 3, "bgra8", bgra.tobytes(), channels=4))
    assert out.shape == (3, 3, 3)
    assert (out[..., 0] == 1).all() and (out[..., 2] == 3).all()


def test_mono8_expands_to_three_channels():
    mono = np.array([[7, 8], [9, 10]], dtype=np.uint8)
    out = image_msg_to_bgr(FakeImage(2, 2, "mono8", mono.tobytes(), channels=1))
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[..., 0], mono)
    assert np.array_equal(out[..., 2], mono)


def test_row_padding_is_honoured():
    # 3 px wide bgr8 = 9 bytes/row, padded to a step of 12. Ignoring step
    # shears the image progressively -- row 1 would start 3 bytes early.
    h, w = 3, 3
    step = 12
    img = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    padded = np.zeros((h, step), dtype=np.uint8)
    padded[:, :w * 3] = img.reshape(h, w * 3)
    out = image_msg_to_bgr(FakeImage(h, w, "bgr8", padded.tobytes(), step=step))
    assert np.array_equal(out, img)


def test_unsupported_encoding_raises():
    with pytest.raises(ValueError, match="unsupported Image encoding"):
        image_msg_to_bgr(FakeImage(1, 1, "yuv422", b"\x00" * 4))


def test_truncated_image_raises_rather_than_reshaping_garbage():
    with pytest.raises(ValueError, match="too short"):
        image_msg_to_bgr(FakeImage(4, 4, "bgr8", b"\x00" * 10))


def test_xyz_only_cloud():
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, 0.25]], dtype=np.float32)
    msg = FakeCloud([FakeField("x", 0), FakeField("y", 4), FakeField("z", 8)],
                    12, pts.tobytes(), width=2)
    assert np.allclose(pointcloud2_to_xyz(msg), pts.astype(np.float64))


def test_fields_read_by_offset_not_by_position():
    # A real lidar driver publishes x,y,z,intensity,ring -- 20-byte points.
    # Assuming a packed 12-byte xyz layout would read intensity as the next
    # point's x and produce a plausible-looking but wrong cloud.
    n = 4
    point_step = 20
    raw = np.zeros((n, point_step), dtype=np.uint8)
    xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=np.float32)
    intensity = np.array([99.0, 98.0, 97.0, 96.0], dtype=np.float32)
    ring = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    raw[:, 0:12] = xyz.view(np.uint8).reshape(n, 12)
    raw[:, 12:16] = intensity.view(np.uint8).reshape(n, 4)
    raw[:, 16:20] = ring.view(np.uint8).reshape(n, 4)
    msg = FakeCloud([FakeField("x", 0), FakeField("y", 4), FakeField("z", 8),
                     FakeField("intensity", 12), FakeField("ring", 16)],
                    point_step, raw.tobytes(), width=n)
    assert np.allclose(pointcloud2_to_xyz(msg), xyz.astype(np.float64))


def test_out_of_order_fields():
    # nothing requires x,y,z to be the first three fields or in order
    n = 2
    raw = np.zeros((n, 12), dtype=np.uint8)
    z = np.array([9.0, 10.0], dtype=np.float32)
    x = np.array([1.0, 2.0], dtype=np.float32)
    y = np.array([5.0, 6.0], dtype=np.float32)
    raw[:, 0:4] = z.view(np.uint8).reshape(n, 4)
    raw[:, 4:8] = x.view(np.uint8).reshape(n, 4)
    raw[:, 8:12] = y.view(np.uint8).reshape(n, 4)
    msg = FakeCloud([FakeField("z", 0), FakeField("x", 4), FakeField("y", 8)],
                    12, raw.tobytes(), width=n)
    out = pointcloud2_to_xyz(msg)
    assert np.allclose(out[:, 0], x) and np.allclose(out[:, 1], y) and np.allclose(out[:, 2], z)


def test_float64_cloud():
    pts = np.array([[1.5, -2.5, 3.5]], dtype=np.float64)
    msg = FakeCloud([FakeField("x", 0, 8), FakeField("y", 8, 8), FakeField("z", 16, 8)],
                    24, pts.tobytes(), width=1)
    assert np.allclose(pointcloud2_to_xyz(msg), pts)


def test_empty_cloud_returns_empty_not_error():
    msg = FakeCloud([FakeField("x", 0), FakeField("y", 4), FakeField("z", 8)],
                    12, b"", width=0)
    assert pointcloud2_to_xyz(msg).shape == (0, 3)


def test_missing_field_raises():
    msg = FakeCloud([FakeField("x", 0), FakeField("y", 4)], 8, b"\x00" * 8, width=1)
    with pytest.raises(ValueError, match="missing field"):
        pointcloud2_to_xyz(msg)


def test_integer_xyz_rejected():
    msg = FakeCloud([FakeField("x", 0, 4), FakeField("y", 4, 4), FakeField("z", 8, 4)],
                    12, b"\x00" * 12, width=1)
    with pytest.raises(ValueError, match="non-float datatype"):
        pointcloud2_to_xyz(msg)


def test_bigendian_rejected():
    msg = FakeCloud([FakeField("x", 0), FakeField("y", 4), FakeField("z", 8)],
                    12, b"\x00" * 12, width=1, is_bigendian=True)
    with pytest.raises(ValueError, match="big-endian"):
        pointcloud2_to_xyz(msg)
