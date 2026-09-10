import numpy as np
from perception_costmap.temporal import TemporalObstacleFilter
from perception_costmap.occupancy import GridSpec


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


def test_motion_compensation_moves_world_fixed_obstacle_backwards():
    grid = GridSpec(x_min=-2, x_max=3, y_min=-2, y_max=3, resolution=1.0)
    f = TemporalObstacleFilter((grid.height, grid.width))
    f.conf[2, 3] = 1.0  # obstacle at x=1.5 in the previous robot frame
    f.compensate_motion((0, 0, 0), (1, 0, 0), grid)
    # Robot moved +1 m, so the stationary obstacle is now one cell behind.
    assert f.conf[2, 2] > 0.99
    assert f.conf[2, 3] < 0.01


def test_motion_compensation_rotates_world_fixed_obstacle():
    grid = GridSpec(x_min=-2, x_max=3, y_min=-2, y_max=3, resolution=1.0)
    f = TemporalObstacleFilter((grid.height, grid.width))
    f.conf[2, 3] = 1.0  # x=1.5, y=0.5
    f.compensate_motion((0, 0, 0), (0, 0, np.pi / 2), grid)
    # A +90 degree robot rotation moves the old forward point to robot-right.
    assert f.conf[0:2, 2:4].max() > 0.9


def test_reset_forgets_confirmed_obstacles():
    """A confirmed obstacle must not survive a reset.

    IGVC requires each run start with nothing carried over. The filter is the
    only thing in the perception node that accumulates across ticks.
    """
    f = TemporalObstacleFilter((3, 3))
    obs, seen = _masks((3, 3), [(1, 1)])
    f.update(obs, seen)
    assert f.update(obs, seen)[1, 1]           # confirmed lethal

    assert f.reset() == 1                       # reports what it cleared
    assert f.conf.max() == 0.0

    # and the evidence is genuinely gone: one hit must not re-confirm it,
    # the same two-hit rule a cold-started filter enforces
    assert not f.update(obs, seen)[1, 1]
    assert f.update(obs, seen)[1, 1]


def test_reset_on_a_cold_filter_is_a_no_op():
    f = TemporalObstacleFilter((3, 3))
    assert f.reset() == 0
    assert f.conf.max() == 0.0


def test_reset_does_not_reallocate_the_array():
    """compensate_motion rebinds self.conf; reset must not, or a caller
    holding a reference would keep writing into the abandoned array."""
    f = TemporalObstacleFilter((3, 3))
    before = f.conf
    f.update(*_masks((3, 3), [(0, 0)]))
    f.reset()
    assert f.conf is before
    assert f.conf.dtype == np.float32
