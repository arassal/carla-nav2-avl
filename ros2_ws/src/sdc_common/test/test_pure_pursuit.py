import math
from sdc_common.pure_pursuit import steer_to_target


def test_straight_ahead_zero_steer():
    assert math.isclose(steer_to_target(10.0, 0.0, wheelbase=2.8, max_steer=1.0),
                        0.0, abs_tol=1e-9)


def test_target_left_positive_steer():
    s = steer_to_target(5.0, 5.0, wheelbase=2.8, max_steer=1.0)
    assert s > 0.0


def test_steer_clamped():
    s = steer_to_target(0.1, 5.0, wheelbase=2.8, max_steer=0.5)
    assert -0.5 <= s <= 0.5
