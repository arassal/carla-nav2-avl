"""Offline fusion tests: no model weights or GPU needed."""

import numpy as np
import pytest

from driving_seg.config import CLASSES, MODELS, classes_for_model, paint_order
from driving_seg import fusion
from driving_seg.pipeline import Pipeline


def frame(h=60, w=80):
    return np.full((h, w, 3), 100, np.uint8)


def test_config_integrity():
    prios = [c["priority"] for c in CLASSES.values()]
    assert len(set(prios)) == len(prios)                 # unique priorities
    colors = [c["color"] for c in CLASSES.values()]
    assert len(set(colors)) == len(colors)               # unique colors
    for c in CLASSES.values():
        assert c["model"] in MODELS
    routed = sum(len(classes_for_model(m)) for m in MODELS)
    assert routed == len(CLASSES)                        # every class routed
    assert paint_order()[0] == "road"
    assert paint_order()[-1] == "person"


def test_asymmetric_placement():
    """A mask on the LEFT third must colorize the LEFT third (mirror guard)."""
    f = frame()
    m = np.zeros((60, 80), bool)
    m[:, :20] = True
    out = fusion.compose(f, {"road": m})
    left = out[:, :20].astype(int) - 100
    right = out[:, 60:].astype(int) - 100
    assert np.abs(left).sum() > 0                        # left changed
    assert np.abs(right).sum() == 0                      # right untouched


def test_priority_person_beats_road():
    f = frame()
    full = np.ones((60, 80), bool)
    out = fusion.compose(f, {"road": full, "person": full})
    # interior pixel (away from contour) must be person-red-ish, not green
    px = out[30, 40].astype(int)
    pr, rd = CLASSES["person"]["color"], CLASSES["road"]["color"]
    d_person = sum(abs(px[i] - (0.55 * 100 + 0.45 * pr[i])) for i in range(3))
    d_road = sum(abs(px[i] - (0.55 * 100 + 0.45 * rd[i])) for i in range(3))
    assert d_person < d_road


def test_mask_shape_and_dtype_coercion():
    f = frame()
    m = np.zeros((30, 40), np.uint8)                     # wrong size + dtype
    m[10:20, 5:15] = 255
    out = fusion.compose(f, {"cone": m})
    assert out.shape == f.shape
    assert not np.array_equal(out, f)


def test_unknown_class_ignored():
    f = frame()
    out = fusion.compose(f, {"dragon": np.ones((60, 80), bool)})
    assert np.array_equal(out, f)


class Stub:
    def __init__(self, name, result):
        self.name, self._r = name, result

    def predict(self, bgr):
        return self._r


def test_pipeline_merge_and_timing():
    h, w = 60, 80
    a = np.zeros((h, w), bool); a[:, :10] = True
    b = np.zeros((h, w), bool); b[:, -10:] = True
    p = Pipeline([Stub("scene", {"person": a}),
                  Stub("road", {"road": b}),
                  Stub("course", {})])
    masks = p.masks(frame())
    assert set(masks) == {"person", "road"}
    out, _ = p.overlay(frame())
    assert out.shape == (h, w, 3)
    assert {"scene", "road", "course", "total"} <= set(p.timings)


def test_dead_model_contributes_nothing():
    from driving_seg.models.course import CourseModel
    m = CourseModel(weights="/nonexistent/course.pt")
    assert m.predict(frame()) == {}
    assert m.predict(frame()) == {}                      # warns once, stays dead
