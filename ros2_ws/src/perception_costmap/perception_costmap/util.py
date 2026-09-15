"""
util.py — small pure helpers shared by the node and tools.
"""


def stamp_to_sec(stamp):
    """builtin_interfaces/Time (or anything with .sec/.nanosec) -> float seconds."""
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def is_fresh(stamp_sec, now_sec, max_age):
    """True if a sample stamped at stamp_sec is younger than max_age at now_sec."""
    return (now_sec - stamp_sec) <= max_age


def clear_sample_buffer(buffer):
    """Empty a timestamped sample buffer. True if it cleared.

    ``sample_buffer.TimestampedBuffer`` keeps its history in a ``.samples``
    deque and has no ``clear()`` of its own, so that is what gets emptied. A
    buffer that does grow a ``clear()`` is cleared through it instead, and
    anything else is left untouched and reported rather than guessed at.

    Not load-bearing: every read from these buffers already goes through a
    freshness or sync tolerance, so stale samples cannot be used by a later
    run even if they survive. This just makes a reset mean what it says.
    """
    if hasattr(buffer, "clear"):
        buffer.clear()
        return True
    samples = getattr(buffer, "samples", None)
    if samples is not None and hasattr(samples, "clear"):
        samples.clear()
        return True
    return False
