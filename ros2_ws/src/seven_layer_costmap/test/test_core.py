import numpy as np
import unittest

from seven_layer_costmap.core import (
    GridSpec, TemporalVoxelGrid, fuse_layers, inflate, infer_road_condition,
    rasterize_predictions,
)


class CoreTests(unittest.TestCase):
    def test_fusion_uses_maximum_cost(self):
        layers = {'lanelet': np.array([[10, 80]], dtype=np.uint8),
                  'static_obstacle': np.array([[100, 0]], dtype=np.uint8)}
        self.assertEqual(fuse_layers(layers).tolist(), [[100, 80]])

    def test_fusion_rejects_geometry_mismatch(self):
        with self.assertRaises(ValueError):
            fuse_layers({'lanelet': np.zeros((2, 2)), 'inflation': np.zeros((3, 3))})

    def test_inflation_is_lethal_at_source_and_decays(self):
        source = np.zeros((9, 9), dtype=np.uint8)
        source[4, 4] = 100
        result = inflate(source, radius_m=2.0, resolution=1.0, decay=1.0)
        self.assertEqual(result[4, 4], 100)
        self.assertTrue(0 < result[4, 5] < 99)
        self.assertEqual(result[0, 0], 0)

    def test_temporal_voxels_expire(self):
        grid = TemporalVoxelGrid(GridSpec(10, 10, 1), persistence_s=2.0)
        grid.observe(np.array([[0.0, 0.0, 1.0]]), stamp_s=10.0)
        self.assertEqual(grid.project(11.9).max(), 100)
        self.assertEqual(grid.project(12.1).max(), 0)

    def test_prediction_projects_motion_forward(self):
        result = rasterize_predictions(GridSpec(20, 20, 1), [(0, 0, 2, 0)], horizons=(1.0,))
        self.assertGreater(result[10, 12], 0)

    def test_road_condition_dry_and_low_visibility(self):
        dry = np.zeros((20, 20, 3), dtype=np.uint8) + np.array([70, 90, 110], dtype=np.uint8)
        dark = np.full((20, 20, 3), 10, dtype=np.uint8)
        self.assertEqual(infer_road_condition(dry)[0], 'dry')
        self.assertEqual(infer_road_condition(dark)[0], 'low_visibility')
