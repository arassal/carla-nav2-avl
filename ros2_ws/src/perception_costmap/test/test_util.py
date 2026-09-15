from types import SimpleNamespace
from perception_costmap.util import stamp_to_sec, is_fresh


def test_stamp_to_sec():
    stamp = SimpleNamespace(sec=12, nanosec=500_000_000)
    assert abs(stamp_to_sec(stamp) - 12.5) < 1e-9


def test_is_fresh_within_budget():
    assert is_fresh(stamp_sec=10.0, now_sec=10.3, max_age=0.5)


def test_is_stale_past_budget():
    assert not is_fresh(stamp_sec=10.0, now_sec=10.6, max_age=0.5)


def test_clear_sample_buffer_handles_both_known_shapes():
    from perception_costmap.util import clear_sample_buffer

    class WithClear:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    class WithSamples:
        def __init__(self):
            self.samples = [(1.0, "a"), (2.0, "b")]

    a = WithClear()
    assert clear_sample_buffer(a) is True and a.cleared

    b = WithSamples()
    assert clear_sample_buffer(b) is True and b.samples == []


def test_clear_sample_buffer_leaves_unknown_shapes_alone():
    """Guess nothing about an API we cannot see -- report failure instead."""
    from perception_costmap.util import clear_sample_buffer

    class Opaque:
        pass

    assert clear_sample_buffer(Opaque()) is False
