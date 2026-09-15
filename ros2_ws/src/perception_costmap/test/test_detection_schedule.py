from perception_costmap.detection_schedule import DetectionScheduler


def test_primary_runs_every_tick_and_sides_rotate():
    scheduler = DetectionScheduler(primary="front", secondary_stride=1)
    assert scheduler.select(["front", "left", "right"]) == ["front", "left"]
    assert scheduler.select(["front", "left", "right"]) == ["front", "right"]
    assert scheduler.select(["front", "left", "right"]) == ["front", "left"]


def test_stride_bounds_secondary_load():
    scheduler = DetectionScheduler(primary="front", secondary_stride=2)
    selections = [scheduler.select(["front", "left", "right"])
                  for _ in range(4)]
    assert selections == [
        ["front", "left"], ["front"],
        ["front", "right"], ["front"],
    ]


def test_missing_primary_still_rotates_available_cameras():
    scheduler = DetectionScheduler(primary="front")
    assert scheduler.select(["left", "right"]) == ["left"]
    assert scheduler.select(["left", "right"]) == ["right"]
