from perception_costmap.health import ERROR, OK, WARN, camera_health


def test_camera_health_requires_current_rgb_and_depth():
    assert camera_health(0.1, 0.1, None, 0.5) == (
        OK, "RGB and ZED depth current")
    assert camera_health(0.8, 0.1, None, 0.5)[0] == ERROR
    assert camera_health(0.1, None, None, 0.5)[0] == ERROR


def test_missing_expected_confidence_is_warning_not_fake_depth_failure():
    assert camera_health(0.1, 0.1, None, 0.5, True)[0] == WARN
