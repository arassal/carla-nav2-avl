"""
Tests for the fusion logic in costmap_node.CostmapNode._tick. This is the
one ROS-adjacent module in the package, but __init__/_tick are plain Python
(no rclpy import happens unless attach_ros()/main() are called), so we can
drive it directly with synthetic camera images and lidar points.
"""

import numpy as np

from perception_costmap.occupancy import GridSpec, FREE, LETHAL, UNKNOWN
from perception_costmap.bev import homography_from_camera
from perception_costmap.costmap_node import CostmapNode, CameraSource


def _make_forward_camera(grid, image_shape=(120, 160)):
    K = np.array([[150, 0, 80], [0, 150, 60], [0, 0, 1]], dtype=np.float64)
    H = homography_from_camera(K, cam_height=1.5, pitch_deg=25, grid=grid)
    return CameraSource("front", H, grid, image_shape, max_age=1.0)


def test_tick_with_no_fresh_sensors_is_all_unknown():
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=False)
    cost = node._tick(now_sec=100.0)
    assert (cost == UNKNOWN).all()


def test_tick_uses_fresh_camera_image():
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    img = np.full((120, 160, 3), 128, dtype=np.uint8)   # uniform "road" grey
    cam.on_image(img, stamp_sec=100.0)
    node = CostmapNode(grid, {"front": cam}, segmentation_method="hsv",
                       obstacle_method="classical", temporal_enabled=False)
    cost = node._tick(now_sec=100.05)
    # some cells within the camera's known footprint should now be FREE
    assert (cost == FREE).any()
    # cells outside the camera's ground footprint should stay UNKNOWN
    assert (cost == UNKNOWN).any()


def test_tick_ignores_stale_camera():
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid, image_shape=(120, 160))
    img = np.full((120, 160, 3), 128, dtype=np.uint8)
    cam.on_image(img, stamp_sec=0.0)   # ancient stamp
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=False)
    cost = node._tick(now_sec=100.0)   # 100s later, way past max_age=1.0
    assert (cost == UNKNOWN).all()


def test_tick_lidar_only_publishes_obstacles_without_marking_road():
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=False)
    pts = np.array([[3.0, 0.0, 0.5]])   # one obstacle point in front
    node.on_lidar(pts, stamp_sec=100.0)
    cost = node._tick(now_sec=100.1)
    col, row = grid.world_to_cell(3.0, 0.0)
    assert cost[row, col] == LETHAL
    # camera never contributed -> no FREE road cells anywhere
    assert not (cost == FREE).any()


def test_going_blind_clears_previously_latched_obstacles():
    """
    Every other 'no fresh sensors' test above starts from a virgin node, so
    the temporal filter has nothing latched and they pass even when a blind
    tick republishes stale belief. This one establishes a confident obstacle
    FIRST, then goes blind -- which is what actually happened against CARLA
    (6.69% of cells stayed LETHAL, bit identical, 40s after the feed died).
    """
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=True,
                       temporal_kw=dict(hit=0.4, miss=0.2, threshold=0.5))
    pts = np.array([[3.0, 0.0, 0.5]])
    col, row = grid.world_to_cell(3.0, 0.0)

    node.on_lidar(pts, stamp_sec=100.0)
    node._tick(now_sec=100.1)
    node.on_lidar(pts, stamp_sec=100.2)
    assert node._tick(now_sec=100.3)[row, col] == LETHAL   # latched

    # now every sensor stops: max_age is 0.5s, so 10s later nothing is fresh
    blind = node._tick(now_sec=110.0)
    assert (blind == UNKNOWN).all()
    # and it must stay unknown, not drift back
    assert (node._tick(now_sec=120.0) == UNKNOWN).all()


def test_recovery_after_blindness_keeps_temporal_history():
    """A dropout must not cost us the confidence history -- one fresh lidar
    frame after recovery should re-confirm, not restart from zero."""
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=True,
                       temporal_kw=dict(hit=0.4, miss=0.2, threshold=0.5))
    pts = np.array([[3.0, 0.0, 0.5]])
    col, row = grid.world_to_cell(3.0, 0.0)

    node.on_lidar(pts, stamp_sec=100.0)
    node._tick(now_sec=100.1)
    node.on_lidar(pts, stamp_sec=100.2)
    node._tick(now_sec=100.3)

    assert (node._tick(now_sec=110.0) == UNKNOWN).all()    # blind

    node.on_lidar(pts, stamp_sec=110.5)
    assert node._tick(now_sec=110.6)[row, col] == LETHAL


def test_tick_temporal_filter_suppresses_single_frame_obstacle():
    grid = GridSpec(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    cam = _make_forward_camera(grid)
    node = CostmapNode(grid, {"front": cam}, temporal_enabled=True,
                       temporal_kw=dict(hit=0.4, miss=0.2, threshold=0.5))
    pts = np.array([[3.0, 0.0, 0.5]])
    node.on_lidar(pts, stamp_sec=100.0)
    cost1 = node._tick(now_sec=100.1)
    col, row = grid.world_to_cell(3.0, 0.0)
    assert cost1[row, col] != LETHAL     # single hit (0.4) below threshold
    node.on_lidar(pts, stamp_sec=100.2)
    cost2 = node._tick(now_sec=100.3)
    assert cost2[row, col] == LETHAL     # second hit crosses threshold
