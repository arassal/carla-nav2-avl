"""
util.py — pure stamp/staleness helpers plus message-buffer decoding. No ROS
imports: costmap_node.py and tools/carla_feed.py both need "is this sensor
data too old to trust", and that logic is exactly the #1 sim-to-real bug
(silently building a costmap from a frozen frame after a camera dies).
Keeping it here means it's unit-testable without a ROS graph.

The Image/PointCloud2 decoders below are here for the same reason. They take
duck-typed messages (anything with the right attributes), so the subscription
path in costmap_node is exercised by the offline suite instead of only ever
being tested by running the real node. They also keep cv_bridge off the
dependency list, which matters on the Jetson where cv_bridge and the system
OpenCV disagree about numpy versions often enough to be a bring-up hazard.
"""

import numpy as np


def stamp_to_sec(stamp) -> float:
    """Convert a ROS-style stamp (has .sec / .nanosec, or is already a
    float/int) to seconds as a float. Accepts builtin_interfaces/Time,
    a SimpleNamespace with the same fields (for tests), or a plain number."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", 0)
    if sec is None:
        raise TypeError(f"stamp {stamp!r} has no .sec and is not numeric")
    return float(sec) + float(nanosec) * 1e-9


def is_fresh(stamp_sec: float, now_sec: float, max_age: float) -> bool:
    """True iff the sensor reading is not older than max_age seconds.
    Also rejects stamps from the future beyond a small tolerance (clock
    jumps / sim-time resets should read as stale, not as fresh forever)."""
    age = now_sec - stamp_sec
    return -0.5 <= age <= max_age


# Channel count per sensor_msgs/Image encoding, and where B, G, R sit in it.
_ENCODINGS = {
    "bgr8": (3, (0, 1, 2)),
    "rgb8": (3, (2, 1, 0)),
    "bgra8": (4, (0, 1, 2)),
    "rgba8": (4, (2, 1, 0)),
    "mono8": (1, (0, 0, 0)),
}


def image_msg_to_bgr(msg):
    """
    sensor_msgs/Image -> (H, W, 3) uint8 BGR, the layout every module in this
    package expects. Duck-typed: anything with height/width/encoding/step/data.

    `step` is honoured rather than assumed to equal width*channels -- a
    driver that row-pads (or an image whose width is not a multiple of the
    alignment) otherwise decodes as a progressively sheared picture, which
    looks like a broken camera rather than a broken reader.
    """
    enc = msg.encoding
    if enc not in _ENCODINGS:
        raise ValueError(
            "unsupported Image encoding %r (supported: %s). Republish in one "
            "of these or extend _ENCODINGS." % (enc, ", ".join(sorted(_ENCODINGS)))
        )
    channels, (bi, gi, ri) = _ENCODINGS[enc]
    h, w = int(msg.height), int(msg.width)
    step = int(msg.step) or w * channels
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    expected = step * h
    if buf.size < expected:
        raise ValueError("Image data too short: got %d bytes, need %d (%dx%d, "
                         "step %d)" % (buf.size, expected, w, h, step))
    rows = buf[:expected].reshape(h, step)
    img = rows[:, :w * channels].reshape(h, w, channels)
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    return np.stack((img[:, :, bi], img[:, :, gi], img[:, :, ri]), axis=2)


# sensor_msgs/PointField datatype ids -> numpy dtype. Only the float types
# are accepted for x/y/z; an integer x/y/z field means the producer is using
# some scaled encoding this reader would silently misinterpret.
_PF_DTYPES = {7: np.float32, 8: np.float64}


def pointcloud2_to_xyz(msg):
    """
    sensor_msgs/PointCloud2 -> (N, 3) float64 array of x, y, z.

    Reads the x/y/z fields by their declared offsets instead of assuming the
    common 12-byte-xyz layout, so a cloud that also carries intensity/ring
    (every real lidar driver) decodes correctly rather than reading garbage
    out of the neighbouring fields.
    """
    if getattr(msg, "is_bigendian", False):
        raise ValueError("big-endian PointCloud2 is not supported")
    by_name = {f.name: f for f in msg.fields}
    missing = [n for n in ("x", "y", "z") if n not in by_name]
    if missing:
        raise ValueError("PointCloud2 is missing field(s) %s; has %s"
                         % (missing, sorted(by_name)))

    point_step = int(msg.point_step)
    n = int(msg.width) * int(msg.height)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if n == 0 or buf.size < n * point_step:
        n = buf.size // point_step if point_step else 0
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)
    raw = buf[:n * point_step].reshape(n, point_step)

    out = np.empty((n, 3), dtype=np.float64)
    for i, name in enumerate(("x", "y", "z")):
        f = by_name[name]
        dt = _PF_DTYPES.get(int(f.datatype))
        if dt is None:
            raise ValueError("PointCloud2 field %r has non-float datatype %d"
                             % (name, f.datatype))
        width = np.dtype(dt).itemsize
        off = int(f.offset)
        if off + width > point_step:
            raise ValueError("PointCloud2 field %r (offset %d, %d bytes) runs "
                             "past point_step %d" % (name, off, width, point_step))
        out[:, i] = raw[:, off:off + width].copy().view(dt).reshape(n)
    return out
