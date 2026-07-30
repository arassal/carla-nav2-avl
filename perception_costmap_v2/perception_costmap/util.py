"""
util.py — pure stamp/staleness helpers. No ROS imports: costmap_node.py and
tools/carla_feed.py both need "is this sensor data too old to trust", and
that logic is exactly the #1 sim-to-real bug (silently building a costmap
from a frozen frame after a camera dies). Keeping it here means it's
unit-testable without a ROS graph.
"""


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
