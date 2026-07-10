"""
occupancy.py
------------
Turn grid-space masks (road / obstacle / observed) into a ROS
``nav_msgs/OccupancyGrid``.

This is the heart of the costmap and is deliberately ROS-free at its core so
it can be unit-tested without a running ROS graph. ``build_cost_array`` is pure
numpy; ``to_occupancy_grid_msg`` imports ROS message types lazily.

Conventions
-----------
- Robot-centric grid, REP-103 axes: +x forward, +y left.
- The grid array is shaped (height, width) = (rows along y, cols along x),
  matching ``OccupancyGrid`` semantics where index = row * width + col,
  column indexes world x and row indexes world y.
- Cost values follow the ROS convention:
      -1  unknown (not observed yet)
       0  free / drivable road
     100  lethal (a detected obstacle, or the inflation core at distance 0)
  Between FREE and LETHAL, cells near an obstacle carry an intermediate cost
  that decays with distance (see ``inflate_costs``), and off-road-but-no-
  obstacle cells carry a separate, tunable ``offroad_cost`` -- distinct from
  LETHAL so a planner can tell "don't leave the lane" from "something is
  actually there."
"""

from dataclasses import dataclass
import numpy as np
import cv2

UNKNOWN = -1
FREE = 0
LETHAL = 100
# Off-road (observed, not drivable, no confirmed obstacle) used to collapse
# into LETHAL by default, which made every non-road pixel look identical to
# a real obstacle in rviz. Distinct default so the two are visually and
# numerically different -- still high enough to be avoided.
DEFAULT_OFFROAD_COST = 65
DEFAULT_INFLATION_RADIUS = 0.8   # metres
DEFAULT_COST_SCALING_FACTOR = 4.0


@dataclass(frozen=True)
class GridSpec:
    """Metric extent + resolution of the costmap, in the robot frame."""
    x_min: float = -4.0      # metres behind the robot
    x_max: float = 16.0      # metres ahead of the robot
    y_min: float = -10.0     # metres to the right (+y is left)
    y_max: float = 10.0      # metres to the left
    resolution: float = 0.1  # metres per cell
    frame_id: str = "base_link"

    @property
    def width(self) -> int:          # cells along x (forward)
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:         # cells along y (left)
        return int(round((self.y_max - self.y_min) / self.resolution))

    def world_to_cell(self, x: float, y: float):
        """World (x,y) in metres -> (col, row), or None if outside the grid."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        if 0 <= col < self.width and 0 <= row < self.height:
            return col, row
        return None

    def cell_to_world(self, col: int, row: int):
        """(col,row) -> world (x,y) at the cell centre, in metres."""
        x = self.x_min + (col + 0.5) * self.resolution
        y = self.y_min + (row + 0.5) * self.resolution
        return x, y


def inflate_costs(obstacle_mask: np.ndarray, resolution: float,
                  inflation_radius: float = DEFAULT_INFLATION_RADIUS,
                  cost_scaling_factor: float = DEFAULT_COST_SCALING_FACTOR,
                  lethal: int = LETHAL) -> np.ndarray:
    """
    Nav2-style exponential-decay halo around obstacle cells:

        cost(d) = lethal * exp(-cost_scaling_factor * d)   for 0 <= d <= inflation_radius
        cost(d) = NO_INFLUENCE                              for d > inflation_radius

    ``d`` is the Euclidean distance in metres from the nearest True cell in
    ``obstacle_mask``, computed with ``cv2.distanceTransform`` (fast, no
    scipy dependency -- this module stays torch/scipy-free by design).

    Returns a float array the same shape as ``obstacle_mask``. Cells beyond
    ``inflation_radius`` (or when there are no obstacles at all) get -1.0,
    a sentinel that is a no-op under ``np.maximum`` against any real cost
    (including UNKNOWN, also -1) -- callers combine with
    ``np.maximum(existing_cost, inflated)``.
    """
    shape = obstacle_mask.shape
    if not obstacle_mask.any():
        return np.full(shape, -1.0, dtype=np.float64)

    # distanceTransform wants uint8 with 0 = feature point (the obstacle);
    # everything else is measured as distance-to-nearest-zero.
    not_obstacle = (~obstacle_mask.astype(bool)).astype(np.uint8)
    dist_px = cv2.distanceTransform(not_obstacle, cv2.DIST_L2, 5)
    dist_m = dist_px * resolution

    decay = lethal * np.exp(-cost_scaling_factor * dist_m)
    decay[dist_m > inflation_radius] = -1.0
    decay[obstacle_mask.astype(bool)] = float(lethal)
    return decay


def build_cost_array(grid: GridSpec,
                     road_mask: np.ndarray,
                     obstacle_mask: np.ndarray,
                     known_mask: np.ndarray = None,
                     offroad_cost: int = DEFAULT_OFFROAD_COST,
                     inflation_radius: float = DEFAULT_INFLATION_RADIUS,
                     cost_scaling_factor: float = DEFAULT_COST_SCALING_FACTOR,
                     unknown_cost: int = UNKNOWN) -> np.ndarray:
    """
    Fuse grid-space boolean masks into an int8 cost array (height, width).

    Base layer (low -> high): unknown < off-road < road < obstacle.
      - cells not in ``known_mask``      -> ``unknown_cost`` (default -1)
      - known cells                      -> ``offroad_cost``
      - road cells (within known)        -> FREE (0)
      - obstacle cells                   -> LETHAL (100)

    ``unknown_cost``: -1 keeps ROS "unknown" semantics (planner decides).
    A small positive value (e.g. 25) makes blind spots *traversable with a
    mild penalty* -- the planner prefers observed-free ground but may cross
    a camera blind wedge. Obstacle inflation still bleeds into blind cells
    (see below), so blind ground next to a detected obstacle stays expensive.

    Then an inflation halo is applied around obstacle cells (see
    ``inflate_costs``): any cell within ``inflation_radius`` of an obstacle
    gets at least the decayed cost, even if it was UNKNOWN or FREE -- a real
    obstacle's danger zone isn't gated on camera visibility elsewhere.
    Obstacle cells themselves are always pinned back to exact LETHAL after
    inflation, since the float decay only reaches ~lethal at d=0 by
    approximation, not exactly.

    Road is clipped to ``known_mask``: a "road" pixel outside the observed
    footprint (e.g. a mirror-projected sky pixel from a side camera) must
    never mark unobserved ground drivable. Obstacles are NOT clipped --
    the lidar legitimately sees beyond the camera FOVs, and a spurious
    obstacle is the safe direction.

    All masks must be shape (grid.height, grid.width).
    """
    shape = (grid.height, grid.width)
    for name, m in (("road_mask", road_mask), ("obstacle_mask", obstacle_mask)):
        if m.shape != shape:
            raise ValueError(f"{name} shape {m.shape} != grid {shape}")

    if known_mask is None:
        known_mask = np.ones(shape, dtype=bool)

    cost = np.full(shape, float(unknown_cost), dtype=np.float64)
    cost[known_mask] = float(offroad_cost)      # observed but not road -> off-road
    cost[road_mask.astype(bool) & known_mask] = FREE   # road overrides off-road

    inflated = inflate_costs(obstacle_mask, grid.resolution,
                             inflation_radius, cost_scaling_factor)
    cost = np.maximum(cost, inflated)           # halo only ever raises cost
    cost[obstacle_mask.astype(bool)] = LETHAL   # obstacle cells: exact lethal

    return cost.astype(np.int8)


def to_occupancy_grid_msg(cost: np.ndarray, grid: GridSpec, stamp=None,
                          frame_id: str = None):
    """
    Wrap a cost array in a ``nav_msgs/OccupancyGrid``. ROS imports are lazy so
    the rest of this module stays usable (and testable) without ROS installed.
    """
    from nav_msgs.msg import OccupancyGrid
    from geometry_msgs.msg import Pose

    msg = OccupancyGrid()
    msg.header.frame_id = frame_id or grid.frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = float(grid.resolution)
    msg.info.width = grid.width
    msg.info.height = grid.height
    origin = Pose()
    origin.position.x = float(grid.x_min)
    origin.position.y = float(grid.y_min)
    origin.orientation.w = 1.0
    msg.info.origin = origin
    # OccupancyGrid is row-major (index = row*width + col); numpy C-order
    # flatten of a (height, width) array gives exactly that.
    msg.data = cost.astype(np.int8).flatten().tolist()
    return msg
