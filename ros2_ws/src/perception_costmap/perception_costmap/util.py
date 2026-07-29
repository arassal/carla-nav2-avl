"""
util.py — small pure helpers shared by the node and tools.
"""


def stamp_to_sec(stamp):
    """builtin_interfaces/Time (or anything with .sec/.nanosec) -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def is_fresh(stamp_sec, now_sec, max_age):
    """True if a sample stamped at stamp_sec is younger than max_age at now_sec."""
    return (now_sec - stamp_sec) <= max_age
