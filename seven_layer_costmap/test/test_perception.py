import unittest

import numpy as np

from seven_layer_costmap.core import GridSpec
from seven_layer_costmap.perception import (
    CameraSample, PersistenceSeparator, Pose2D, SkewMonitor, ThreeCameraSynchronizer,
    depth_to_base_points, obstacle_grid, traffic_regulation_layer,
    inflation_radius_for_speed, remove_ground, vision_lane_layer, WorldOccupancyModel,
)


def sample(stamp):
    return CameraSample(np.ones((2, 2)), np.zeros((2, 2, 3), dtype=np.uint8),
                        np.eye(3), stamp)


class PerceptionTests(unittest.TestCase):
    def test_three_camera_sync_accepts_bounded_skew_once(self):
        sync = ThreeCameraSynchronizer(max_skew_s=0.05)
        sync.update('front', sample(10.00))
        sync.update('left', sample(10.02))
        sync.update('right', sample(10.04))
        self.assertIsNotNone(sync.take())
        self.assertIsNone(sync.take())

    def test_three_camera_sync_rejects_excess_skew(self):
        sync = ThreeCameraSynchronizer(max_skew_s=0.05)
        sync.update('front', sample(10.00))
        sync.update('left', sample(10.02))
        sync.update('right', sample(10.20))
        self.assertIsNone(sync.take())

    def test_skew_monitor_counts_violations(self):
        monitor = SkewMonitor(0.05)
        monitor.observe([1.0, 1.02, 1.04])
        monitor.observe([2.0, 2.02, 2.10])
        summary = monitor.summary()
        self.assertEqual(summary['violations'], 1)
        self.assertAlmostEqual(summary['max_s'], 0.10)

    def test_speed_dependent_inflation_is_bounded(self):
        self.assertAlmostEqual(inflation_radius_for_speed(2.0, 4.0, 0.5, 3.0), 4.0)
        self.assertAlmostEqual(inflation_radius_for_speed(2.0, 20.0, 0.5, 3.0), 5.0)

    def test_depth_backprojection_optical_to_base(self):
        depth = np.array([[2.0]], dtype=np.float32)
        k = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        point = depth_to_base_points(depth, k, [1.0, 2.0, 3.0], stride=1)[0]
        np.testing.assert_allclose(point, [3.0, 2.0, 3.0])

    def test_obstacle_height_filter(self):
        spec = GridSpec(10, 10, 1)
        grid = obstacle_grid(spec, [[1, 1, -2.0], [2, 1, 1.0]])
        self.assertEqual(int((grid == 100).sum()), 1)

    def test_dominant_ground_band_is_removed(self):
        ground = np.column_stack((np.arange(100), np.zeros(100), np.full(100, -1.0)))
        obstacle = np.array([[2.0, 0.0, 0.2]])
        result = remove_ground(np.vstack((ground, obstacle)))
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(float(result[0, 2]), 0.2)

    def test_persistence_separates_static(self):
        separator = PersistenceSeparator((3, 3), static_hits=3, decay=0)
        occupied = np.zeros((3, 3), dtype=np.uint8)
        occupied[1, 1] = 100
        separator.update(occupied)
        static, transient = separator.update(occupied)
        self.assertEqual(static[1, 1], 100)
        self.assertEqual(transient[1, 1], 0)

    def test_lane_layer_has_forward_low_cost_corridor(self):
        spec = GridSpec(20, 20, 1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        layer = vision_lane_layer(spec, image)
        self.assertEqual(layer[10, 12], 0)
        self.assertGreater(layer[2, 12], 0)

    def test_red_signal_adds_stop_barrier(self):
        spec = GridSpec(20, 20, 1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:10, :2, 2] = 255
        layer = traffic_regulation_layer(spec, [image])
        self.assertEqual(layer.max(), 100)

    def test_world_occupancy_compensates_vehicle_translation(self):
        spec = GridSpec(20, 20, 1)
        model = WorldOccupancyModel(spec, voxel_m=1.0, z_resolution=1.0)
        model.observe([[5.0, 0.0, 0.0]], Pose2D(), [[0.0, 0.0, 0.0]], 1.0)
        _, before = model.project(Pose2D(), 1.0)
        _, after = model.project(Pose2D(x=2.0), 1.1)
        self.assertEqual(before[10, 15], 100)
        self.assertEqual(after[10, 13], 100)

    def test_world_occupancy_rejects_invalid_limits(self):
        with self.assertRaises(ValueError):
            WorldOccupancyModel(GridSpec(20, 20, 1), max_clear_rays=0)

    def test_world_occupancy_ray_clears_old_obstacle(self):
        spec = GridSpec(30, 30, 1)
        model = WorldOccupancyModel(spec, voxel_m=1.0, z_resolution=1.0)
        model.observe([[5.0, 0.0, 0.0]], Pose2D(), [[0.0, 0.0, 0.0]], 1.0)
        model.observe([[10.0, 0.0, 0.0]], Pose2D(), [[0.0, 0.0, 0.0]], 1.1)
        static, transient = model.project(Pose2D(), 1.1)
        merged = np.maximum(static, transient)
        self.assertEqual(merged[15, 20], 0)
        self.assertEqual(merged[15, 25], 100)


if __name__ == '__main__':
    unittest.main()
