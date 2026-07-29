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


# --- per-class danger zones -------------------------------------------------
# One inflation radius for everything treats a pedestrian and a parked car as
# equally dangerous, which is wrong in both directions: you routinely drive
# within a metre of a car, and you should never do that to a person. Each
# class therefore gets its own (radius, scaling) pair. Larger radius + SMALLER
# scaling = the cost stays high further out (people). Small radius + LARGE
# scaling = cost collapses within centimetres (cars, cones).
DEFAULT_OBSTACLE_CLASSES = {
    "person":  dict(radius=2.5, scaling=1.5),   # wide, slow decay: keep clear
    "vehicle": dict(radius=1.0, scaling=5.0),   # tight: passing close is normal
    "cone":    dict(radius=0.6, scaling=5.0),   # tight
    "generic": dict(radius=0.8, scaling=4.0),   # unclassified blob: old default
}

# Distance (m) over which cost ramps up as you approach the road edge from
# *inside* the road, and how fast. This is what makes "the closer you get to
# leaving the road, the worse it gets" a gradient instead of a cliff.
DEFAULT_ROAD_EDGE_RADIUS = 1.5
DEFAULT_ROAD_EDGE_SCALING = 2.0


def infill_unknown(cost: np.ndarray, known_mask: np.ndarray,
                   resolution: float, prior: float = 25.0,
                   falloff_m: float = 2.0) -> np.ndarray:
    """
    Guess costs for unobserved cells from the surrounding observed ground.

    Telea inpainting propagates the observed cost field into the blind
    region (camera seams, the rear wedge), then the guess is blended back
    toward ``prior`` as distance from the nearest observed cell grows:

        guess(d) = w * inpainted + (1 - w) * prior,   w = exp(-d / falloff_m)

    Right at a seam the guess is dominated by what surrounds it (road on
    both sides of a seam -> the seam is probably road); deep inside the
    rear wedge, where nothing nearby has been seen for metres, w -> 0 and
    the guess honestly degrades to the neutral prior instead of smearing a
    distant observation across ground we know nothing about.

    Returns a float array; caller combines it into the cost field. Only
    unknown cells are guessed -- observed cells pass through untouched.
    """
    unknown = ~known_mask.astype(bool)
    if not unknown.any() or known_mask.sum() == 0:
        return cost

    src_img = np.clip(cost, 0, LETHAL).astype(np.uint8)
    inpainted = cv2.inpaint(src_img, unknown.astype(np.uint8), 3,
                            cv2.INPAINT_TELEA).astype(np.float64)

    dist_m = cv2.distanceTransform(unknown.astype(np.uint8),
                                   cv2.DIST_L2, 5) * resolution
    w = np.exp(-dist_m / max(falloff_m, 1e-6))

    out = cost.copy()
    guess = w * inpainted + (1.0 - w) * float(prior)
    # Preserve obstacle-halo bleed, but NOT the unknown_cost baseline: blind
    # cells all start at exactly `prior`, and clamping the guess against that
    # would forbid ever guessing anything better than the prior (a road seam
    # would stay 25 instead of ~0). Only cost strictly above the prior in a
    # blind cell can have come from a halo -- keep that as a floor.
    halo_floor = np.where(cost > float(prior), cost, 0.0)
    out[unknown] = np.maximum(halo_floor[unknown], guess[unknown])
    return out


def build_cost_array(grid: GridSpec,
                     road_mask: np.ndarray,
                     obstacle_mask: np.ndarray,
                     known_mask: np.ndarray = None,
                     offroad_cost: int = DEFAULT_OFFROAD_COST,
                     inflation_radius: float = DEFAULT_INFLATION_RADIUS,
                     cost_scaling_factor: float = DEFAULT_COST_SCALING_FACTOR,
                     unknown_cost: int = UNKNOWN,
                     obstacle_layers: dict = None,
                     road_edge_radius: float = 0.0,
                     road_edge_scaling: float = DEFAULT_ROAD_EDGE_SCALING,
                     unknown_infill: bool = False,
                     infill_falloff: float = 2.0) -> np.ndarray:
    """
    Fuse grid-space boolean masks into an int8 cost array (height, width).

    Base layer (low -> high): unknown < road < road-edge ramp < off-road < obstacle.
      - cells not in ``known_mask``   -> ``unknown_cost``
      - known, not road               -> ``offroad_cost`` (near-lethal: leaving
                                          the road is a failure, not a shortcut)
      - road cells                    -> FREE (0), then raised by the edge ramp
      - obstacle cells                -> LETHAL (100), plus a per-class halo

    ``road_edge_radius`` > 0 inflates *inward* from the off-road region, so a
    cell 1 m inside the road carries ~offroad*exp(-scaling*1.0) and a cell on
    the boundary carries the full ``offroad_cost``. Without it the road/off-road
    transition is a step, which both plans badly (no incentive to centre in the
    lane) and renders as a flat plateau of colour.

    ``obstacle_layers`` maps a class name -> {"mask": bool array, "radius": m,
    "scaling": float}. Each layer is inflated with its own parameters and
    combined with ``np.maximum``, so overlapping halos take the worst case.
    When omitted, ``obstacle_mask`` is inflated with the single legacy
    ``inflation_radius`` / ``cost_scaling_factor`` (backwards compatible).

    Road is clipped to ``known_mask``: a "road" pixel outside the observed
    footprint must never mark unobserved ground drivable. Obstacles are NOT
    clipped -- a spurious obstacle is the safe direction.
    """
    shape = (grid.height, grid.width)
    for name, m in (("road_mask", road_mask), ("obstacle_mask", obstacle_mask)):
        if m.shape != shape:
            raise ValueError(f"{name} shape {m.shape} != grid {shape}")

    if known_mask is None:
        known_mask = np.ones(shape, dtype=bool)

    road = road_mask.astype(bool) & known_mask
    cost = np.full(shape, float(unknown_cost), dtype=np.float64)
    cost[known_mask] = float(offroad_cost)      # observed but not road
    cost[road] = FREE                           # road overrides off-road

    # Ramp up as we approach the road edge from inside the lane. Peak is
    # offroad_cost (not LETHAL) so the ramp is continuous with the off-road
    # region it grows out of -- no discontinuity at the boundary.
    if road_edge_radius > 0:
        offroad_region = known_mask & ~road
        if offroad_region.any():
            edge = inflate_costs(offroad_region, grid.resolution,
                                 road_edge_radius, road_edge_scaling,
                                 lethal=int(offroad_cost))
            # Clip the ramp to OBSERVED ground. Without this it bleeds into the
            # blind region and overwrites unknown_cost, silently making blind
            # spots expensive -- the opposite of the documented intent. (Real
            # obstacle halos below are deliberately NOT clipped: a detected
            # obstacle's danger zone is real regardless of what we can see.)
            edge = np.where(known_mask, edge, -1.0)
            cost = np.maximum(cost, edge)

    # Obstacle halos: per-class if given, else the single legacy halo.
    if obstacle_layers:
        for spec in obstacle_layers.values():
            m = spec["mask"]
            if m is None or not m.any():
                continue
            cost = np.maximum(cost, inflate_costs(
                m.astype(bool), grid.resolution,
                spec["radius"], spec["scaling"]))
    else:
        cost = np.maximum(cost, inflate_costs(
            obstacle_mask, grid.resolution,
            inflation_radius, cost_scaling_factor))

    # Blind-spot infill: guess unobserved cells from the surrounding observed
    # ground (see infill_unknown). Runs after the halos so a real obstacle's
    # bleed into the blind region is preserved, before the lethal pin.
    if unknown_infill and unknown_cost >= 0:
        cost = infill_unknown(cost, known_mask, grid.resolution,
                              prior=float(unknown_cost),
                              falloff_m=infill_falloff)

    cost[obstacle_mask.astype(bool)] = LETHAL   # obstacle cells: exact lethal
    return np.clip(cost, -1, LETHAL).astype(np.int8)


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
