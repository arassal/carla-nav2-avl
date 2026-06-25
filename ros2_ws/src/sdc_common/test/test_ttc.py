import numpy as np
from sdc_common.ttc import nearest_frontal_range, decide


def test_nearest_frontal_range_filters_sector():
    pts = np.array([[8.0, 0.0, 0.0], [5.0, 9.0, 0.0], [-3.0, 0.0, 0.0]])
    r = nearest_frontal_range(pts, half_width=1.5, z_lo=-1.0, z_hi=1.0, x_max=50.0)
    assert abs(r - 8.0) < 1e-6


def test_nearest_frontal_range_none_when_clear():
    pts = np.array([[5.0, 9.0, 0.0]])
    assert nearest_frontal_range(pts, 1.5, -1.0, 1.0, 50.0) == float('inf')


def test_decide_states():
    assert decide(float('inf'), speed=5.0, ttc_brake=1.5, ttc_slow=3.0, stop_dist=4.0) == "CLEAR"
    assert decide(3.0, speed=10.0, ttc_brake=1.5, ttc_slow=3.0, stop_dist=4.0) == "EMERGENCY"
    assert decide(3.0, speed=0.1, ttc_brake=1.5, ttc_slow=3.0, stop_dist=4.0) == "EMERGENCY"
    # rng=10, speed=4 -> ttc=2.5, in (ttc_brake=1.5, ttc_slow=3.0] -> SLOW
    assert decide(10.0, speed=4.0, ttc_brake=1.5, ttc_slow=3.0, stop_dist=4.0) == "SLOW"
    # far + slow -> CLEAR
    assert decide(20.0, speed=4.0, ttc_brake=1.5, ttc_slow=3.0, stop_dist=4.0) == "CLEAR"
