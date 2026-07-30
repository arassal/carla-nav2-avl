from types import SimpleNamespace
import pytest

from perception_costmap.util import stamp_to_sec, is_fresh


def test_stamp_to_sec_from_fields():
    stamp = SimpleNamespace(sec=10, nanosec=500_000_000)
    assert stamp_to_sec(stamp) == pytest.approx(10.5)


def test_stamp_to_sec_from_number():
    assert stamp_to_sec(3.25) == 3.25
    assert stamp_to_sec(3) == 3.0


def test_stamp_to_sec_rejects_garbage():
    with pytest.raises(TypeError):
        stamp_to_sec("not a stamp")


def test_is_fresh_within_window():
    assert is_fresh(stamp_sec=10.0, now_sec=10.3, max_age=0.5)


def test_is_fresh_too_old():
    assert not is_fresh(stamp_sec=10.0, now_sec=11.0, max_age=0.5)


def test_is_fresh_future_stamp_small_tolerance_ok():
    # small clock skew (sim-time jitter) should not be treated as stale
    assert is_fresh(stamp_sec=10.4, now_sec=10.0, max_age=0.5)


def test_is_fresh_future_stamp_large_rejected():
    # a stamp far in the future (clock jump / sim reset) must not read as
    # "fresh forever"
    assert not is_fresh(stamp_sec=100.0, now_sec=10.0, max_age=0.5)
