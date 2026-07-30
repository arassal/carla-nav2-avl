import numpy as np
from perception_costmap.temporal import TemporalObstacleFilter


def test_single_frame_does_not_mark_lethal():
    f = TemporalObstacleFilter((5, 5), hit=0.4, miss=0.2, threshold=0.5)
    obs = np.zeros((5, 5), dtype=bool)
    obs[2, 2] = True
    observed = np.ones((5, 5), dtype=bool)
    out = f.update(obs, observed)
    assert not out[2, 2]          # one hit (0.4) is below threshold (0.5)


def test_two_consecutive_hits_marks_lethal():
    f = TemporalObstacleFilter((5, 5), hit=0.4, miss=0.2, threshold=0.5)
    obs = np.zeros((5, 5), dtype=bool)
    obs[2, 2] = True
    observed = np.ones((5, 5), dtype=bool)
    f.update(obs, observed)
    out = f.update(obs, observed)
    assert out[2, 2]              # 0.8 >= 0.5


def test_misses_clear_a_marked_cell():
    f = TemporalObstacleFilter((5, 5), hit=0.4, miss=0.2, threshold=0.5)
    obs = np.zeros((5, 5), dtype=bool)
    obs[2, 2] = True
    observed = np.ones((5, 5), dtype=bool)
    f.update(obs, observed)
    f.update(obs, observed)
    assert f.conf[2, 2] >= 0.5
    empty = np.zeros((5, 5), dtype=bool)
    for _ in range(3):
        f.update(empty, observed)
    assert f.conf[2, 2] < 0.5


def test_unobserved_cells_do_not_decay():
    f = TemporalObstacleFilter((3, 3), hit=0.4, miss=0.2, threshold=0.5)
    obs = np.zeros((3, 3), dtype=bool)
    obs[1, 1] = True
    observed = np.zeros((3, 3), dtype=bool)
    observed[1, 1] = True
    f.update(obs, observed)
    conf_before = f.conf[1, 1]
    # next tick: cell not observed at all (e.g. outside camera FOV, no lidar)
    not_observed = np.zeros((3, 3), dtype=bool)
    f.update(np.zeros((3, 3), dtype=bool), not_observed)
    assert f.conf[1, 1] == conf_before


def test_confidence_clipped_to_unit_interval():
    f = TemporalObstacleFilter((2, 2), hit=0.9, miss=0.9, threshold=0.5)
    obs = np.ones((2, 2), dtype=bool)
    observed = np.ones((2, 2), dtype=bool)
    for _ in range(10):
        f.update(obs, observed)
    assert (f.conf <= 1.0).all() and (f.conf >= 0.0).all()
