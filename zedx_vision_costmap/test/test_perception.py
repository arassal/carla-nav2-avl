import unittest

import numpy as np

from zedx_vision_costmap.core import GridSpec
from zedx_vision_costmap.perception import (
    blind_spot_mask, CameraSample, CentroidTracker, colorize_bev, derive_layers,
    PersistenceSeparator, Pose2D, SkewMonitor, ThreeCameraSynchronizer,
    depth_to_base_points, obstacle_grid, observed_mask_from_rays,
    rotation_matrix_from_rpy,
    inflation_radius_for_speed, remove_ground,
    vision_bev_grid, vision_lane_layer, WorldOccupancyModel,
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

    def test_three_camera_sync_queues_independent_decoder_rates(self):
        sync = ThreeCameraSynchronizer(max_skew_s=0.03, queue_size=10)
        for stamp in (1.00, 1.10, 1.20):
            sync.update('front', sample(stamp))
        for stamp in (1.01, 1.11):
            sync.update('left', sample(stamp))
        sync.update('right', sample(1.02))
        first = sync.take()
        self.assertIsNotNone(first)
        self.assertAlmostEqual(first['front'].stamp_s, 1.00)
        sync.update('right', sample(1.12))
        second = sync.take()
        self.assertIsNotNone(second)
        self.assertAlmostEqual(second['front'].stamp_s, 1.10)

    def test_three_camera_sync_discards_unmatchable_old_heads(self):
        sync = ThreeCameraSynchronizer(max_skew_s=0.03, queue_size=3)
        sync.update('front', sample(1.00))
        sync.update('front', sample(1.10))
        sync.update('left', sample(1.09))
        sync.update('right', sample(1.11))
        matched = sync.take()
        self.assertIsNotNone(matched)
        self.assertAlmostEqual(matched['front'].stamp_s, 1.10)
        self.assertGreaterEqual(sync.dropped, 1)

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

    def test_depth_backprojection_applies_full_mount_rotation(self):
        depth = np.array([[2.0]], dtype=np.float32)
        k = np.eye(3)
        point = depth_to_base_points(
            depth, k, [0.0, 0.0, 0.0], stride=1,
            rpy=[0.0, np.pi / 2, 0.0])[0]
        np.testing.assert_allclose(point, [0.0, 0.0, -2.0], atol=1e-6)
        np.testing.assert_allclose(rotation_matrix_from_rpy([0, 0, 0]), np.eye(3))

    def test_depth_backprojection_rejects_invalid_intrinsics(self):
        with self.assertRaises(ValueError):
            depth_to_base_points(np.ones((2, 2)), np.zeros((3, 3)), [0, 0, 0])

    def test_obstacle_height_filter(self):
        spec = GridSpec(10, 10, 1)
        grid = obstacle_grid(spec, [[1, 1, -2.0], [2, 1, 1.0]])
        self.assertEqual(int((grid == 100).sum()), 1)

    def test_obstacle_grid_can_reject_single_point_noise(self):
        spec = GridSpec(10, 10, 1)
        grid = obstacle_grid(spec, [[1.1, 1.1, 0.5]], min_points_per_cell=2)
        self.assertEqual(int(grid.max()), 0)
        grid = obstacle_grid(
            spec, [[1.1, 1.1, 0.5], [1.2, 1.2, 0.6]], min_points_per_cell=2)
        self.assertEqual(int(grid.max()), 100)

    def test_instantaneous_bev_marks_unknown_free_and_occupied(self):
        spec = GridSpec(20, 20, 1)
        grid, obstacles = vision_bev_grid(
            spec, [[5.0, 0.0, 0.5]], [[0.0, 0.0, 0.0]],
            visibility_max_rays=1, visibility_dilation_cells=0,
            min_points_per_cell=1)
        self.assertEqual(grid[10, 10], 0)
        self.assertEqual(grid[10, 15], 100)
        self.assertEqual(grid[15, 15], -1)
        self.assertEqual(len(obstacles), 1)
        self.assertEqual(colorize_bev(grid).shape, (20, 20, 3))

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

    def test_visibility_mask_follows_depth_ray(self):
        spec = GridSpec(20, 20, 1)
        observed = observed_mask_from_rays(
            spec, [[5.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
            max_rays=1, dilation_cells=0)
        self.assertTrue(observed[10, 10])
        self.assertTrue(observed[10, 15])
        self.assertFalse(observed[15, 15])

    def test_blind_spots_are_only_front_side_wedges(self):
        spec = GridSpec(20, 20, 1)
        blind = blind_spot_mask(spec, half_width_deg=12, min_range_m=1, max_range_m=10)
        self.assertTrue(blind[15, 15])   # approximately +45 degrees
        self.assertTrue(blind[5, 15])    # approximately -45 degrees
        self.assertFalse(blind[10, 15])  # directly forward
        self.assertFalse(blind[15, 5])   # rear-left

    def test_lane_layer_uses_mild_cost_for_unobserved_blind_cell(self):
        spec = GridSpec(20, 20, 1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        observed = np.zeros(spec.shape, dtype=bool)
        layer = vision_lane_layer(spec, image, observed_mask=observed,
                                  blind_half_width_deg=12)
        self.assertEqual(layer[15, 15], 25)
        self.assertEqual(layer[15, 5], 95)
        observed[15, 15] = True
        layer = vision_lane_layer(spec, image, observed_mask=observed,
                                  blind_half_width_deg=12)
        self.assertEqual(layer[15, 15], 0)

    def test_nearby_obstacle_inflation_raises_blind_spot_risk(self):
        spec = GridSpec(20, 20, 1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        occupancy = WorldOccupancyModel(spec, voxel_m=1.0, z_resolution=1.0)
        layers = derive_layers(
            spec, np.array([[5.0, 4.0, 0.5]], dtype=np.float32), image,
            occupancy, CentroidTracker(), Pose2D(), 1.0,
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            inflation_radius_m=2.5, visibility_max_rays=1,
            visibility_dilation_cells=0, blind_half_width_deg=12)
        self.assertEqual(layers['lanelet'][15, 15], 25)
        self.assertGreater(layers['inflation'][15, 15], 25)
        self.assertEqual(layers['spatio_temporal_voxel'][14, 15], 100)

    def test_vision_only_layers_do_not_accumulate_or_predict(self):
        spec = GridSpec(20, 20, 1)
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        occupancy = WorldOccupancyModel(spec, voxel_m=1.0, z_resolution=1.0)
        layers = derive_layers(
            spec, [[5.0, 0.0, 0.5]], image,
            occupancy, CentroidTracker(), Pose2D(), 1.0,
            [[0.0, 0.0, 0.0]], temporal_memory=False,
            enable_prediction=False)
        self.assertFalse(occupancy.last_seen)
        self.assertEqual(int(layers['static_obstacle'].max()), 0)
        self.assertEqual(int(layers['spatio_temporal_voxel'].max()), 100)
        self.assertEqual(int(layers['prediction'].max()), 0)

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
