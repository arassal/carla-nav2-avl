from sdc_common.longitudinal import SpeedPID


def test_accelerates_when_below_target():
    pid = SpeedPID(kp=0.5, ki=0.0, kd=0.0)
    throttle, brake = pid.step(target=10.0, current=0.0, dt=0.1)
    assert throttle > 0.0 and brake == 0.0


def test_brakes_when_above_target():
    pid = SpeedPID(kp=0.5, ki=0.0, kd=0.0)
    throttle, brake = pid.step(target=0.0, current=10.0, dt=0.1)
    assert brake > 0.0 and throttle == 0.0


def test_outputs_clamped_unit_range():
    pid = SpeedPID(kp=100.0, ki=0.0, kd=0.0)
    throttle, brake = pid.step(target=100.0, current=0.0, dt=0.1)
    assert 0.0 <= throttle <= 1.0
