"""ROS-independent health classification for perception streams."""

OK = 0
WARN = 1
ERROR = 2


def camera_health(image_age, depth_age, confidence_age, stale_limit,
                  confidence_expected=False):
    if image_age is None or image_age > stale_limit:
        return ERROR, "RGB stream stale"
    if depth_age is None or depth_age > stale_limit:
        return ERROR, "registered depth stream stale"
    if confidence_expected and (
            confidence_age is None or confidence_age > stale_limit):
        return WARN, "depth confidence stream unavailable"
    return OK, "RGB and ZED depth current"
