from types import SimpleNamespace
from perception_costmap.util import stamp_to_sec, is_fresh


def test_stamp_to_sec():
    stamp = SimpleNamespace(sec=12, nanosec=500_000_000)
    assert abs(stamp_to_sec(stamp) - 12.5) < 1e-9


def test_is_fresh_within_budget():
    assert is_fresh(stamp_sec=10.0, now_sec=10.3, max_age=0.5)


def test_is_stale_past_budget():
    assert not is_fresh(stamp_sec=10.0, now_sec=10.6, max_age=0.5)
