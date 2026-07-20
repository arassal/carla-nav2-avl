import numpy as np
import unittest

from seven_layer_costmap.core import (
    GridSpec, TemporalVoxelGrid, fuse_layers, inflate, rasterize_predictions,
)


class CoreTests(unittest.TestCase):
    def test_grid_spec_rejects_invalid_geometry(self):
        with self.assertRaises(ValueError):
            GridSpec(10, 10, 0)
        with self.assertRaises(ValueError):
            GridSpec(10.1, 10, 1)

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

    def test_inflation_handles_dense_sources_and_zero_radius(self):
        source = np.full((20, 20), 100, dtype=np.uint8)
        self.assertTrue(np.all(inflate(source, 3.0, 0.2) == 100))
        sparse = np.zeros((5, 5), dtype=np.uint8)
        sparse[2, 2] = 100
        zero = inflate(sparse, 0.0, 1.0)
        self.assertEqual(int((zero > 0).sum()), 1)

    def test_inflation_rejects_invalid_parameters(self):
        source = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            inflate(source, -1.0, 1.0)
        with self.assertRaises(ValueError):
            inflate(source, 1.0, 0.0)

    def test_temporal_voxels_expire(self):
        grid = TemporalVoxelGrid(GridSpec(10, 10, 1), persistence_s=2.0)
        grid.observe(np.array([[0.0, 0.0, 1.0]]), stamp_s=10.0)
        self.assertEqual(grid.project(11.9).max(), 100)
        self.assertEqual(grid.project(12.1).max(), 0)

    def test_prediction_projects_motion_forward(self):
        result = rasterize_predictions(GridSpec(20, 20, 1), [(0, 0, 2, 0)], horizons=(1.0,))
        self.assertGreater(result[10, 12], 0)
