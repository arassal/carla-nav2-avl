from pathlib import Path

import yaml


def test_real_car_config_covers_all_cameras_with_depth_and_detectors():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "config/perception_dinosaur.yaml").read_text())
    params = config["perception_costmap"]["ros__parameters"]
    cameras = params["cameras"]
    assert cameras == ["front", "left", "right"]
    assert params["yolo_cameras"] == cameras
    assert params["required_cameras"] == ["front"]
    assert params["cone_cameras"] == ["front"]
    assert params["secondary_detection_stride"] == 2
    assert params["motion_compensation"] is True
    assert params["depth_sync_tolerance_sec"] <= 0.05
    assert params["depth_wait_sec"] < params["image_stale_sec"]
    assert params["depth_confidence_max"] < 95
    assert params["detector_stale_sec"] <= params["image_stale_sec"]
    assert (params["person_exclusion_radius"]
            > params["vehicle_exclusion_radius"]
            > params["generic_exclusion_radius"]
            > params["cone_exclusion_radius"])
    assert params["person_temporal_hit"] >= params["person_temporal_threshold"]
    assert params["vehicle_temporal_hit"] >= params["vehicle_temporal_threshold"]
    assert params["person_temporal_miss"] < params["temporal_miss"]
    for camera in cameras:
        assert params[camera]["depth_topic"].endswith("/depth/depth_registered")
        assert params[camera]["confidence_topic"].endswith(
            "/confidence/confidence_map")
