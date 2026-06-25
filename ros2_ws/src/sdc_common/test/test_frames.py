import math
from sdc_common.frames import carla_xy_to_ros, ros_xy_to_carla, carla_yaw_to_ros


def test_xy_round_trip():
    assert ros_xy_to_carla(*carla_xy_to_ros(10.0, 5.0)) == (10.0, 5.0)


def test_y_is_flipped():
    assert carla_xy_to_ros(3.0, 4.0) == (3.0, -4.0)


def test_yaw_negated_and_wrapped():
    assert math.isclose(carla_yaw_to_ros(math.radians(90)), math.radians(-90), abs_tol=1e-9)
    # -200° carla -> negate -> +200° -> wrap to (-180,180] -> -160°
    assert math.isclose(carla_yaw_to_ros(math.radians(-200)),
                        math.radians(-160), abs_tol=1e-6)
