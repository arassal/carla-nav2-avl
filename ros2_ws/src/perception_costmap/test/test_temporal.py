import numpy as np
from perception_costmap.temporal import TemporalObstacleFilter


def _masks(shape, obstacle_cells):
    obs = np.zeros(shape, bool)
    for r, c in obstacle_cells:
        obs[r, c] = True
    seen = np.ones(shape, bool)
    return obs, seen


def test_needs_two_hits_to_confirm():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    assert not f.update(obs, seen)[1, 1]      # 1st hit: not yet
    assert f.update(obs, seen)[1, 1]          # 2nd hit: confirmed


def test_clears_after_misses():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    f.update(obs, seen); f.update(obs, seen); f.update(obs, seen)  # conf -> 1.0
    empty = np.zeros((3, 3), bool)
    for _ in range(3):
        out = f.update(empty, seen)
    assert not out[1, 1]                      # 1.0 - 3*0.2 = 0.4 < 0.5


def test_unobserved_cells_hold_confidence():
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    f.update(obs, seen); f.update(obs, seen)
    unseen = np.zeros((3, 3), bool)
    out = f.update(np.zeros((3, 3), bool), unseen)   # camera looked away
    assert out[1, 1]                           # still lethal — no evidence it left
