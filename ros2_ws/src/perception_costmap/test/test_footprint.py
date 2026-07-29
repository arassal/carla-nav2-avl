import numpy as np
from perception_costmap.obstacles import boxes_to_footprint_mask


def test_only_bottom_strip_is_marked():
    # box 40 px tall at rows 10..50; frac 0.25 -> bottom 10 rows (40..50)
    m = boxes_to_footprint_mask([(20, 10, 60, 50)], (100, 100, 3), footprint_frac=0.25)
    assert m[45, 30] and m[49, 30]        # inside the strip
    assert not m[20, 30]                  # roof of the box: NOT an obstacle cell
    assert not m[45, 10]                  # left of the box


def test_clips_to_image_and_min_one_row():
    m = boxes_to_footprint_mask([(-5, -5, 8, 3)], (10, 10), footprint_frac=0.1)
    assert m.shape == (10, 10)
    assert m[2, 4]                        # at least one row survives, clipped


def test_empty_boxes():
    assert not boxes_to_footprint_mask([], (5, 5)).any()
