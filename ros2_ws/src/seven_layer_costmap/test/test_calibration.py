from pathlib import Path
import unittest
import yaml


class CalibrationTests(unittest.TestCase):
    def test_sketch_inches_were_converted_to_meters(self):
        path = Path(__file__).parents[1] / 'config' / 'camera_mounts.yaml'
        mounts = yaml.safe_load(path.read_text())['camera_mounts']
        self.assertEqual(mounts['left']['translation_m'], [0.098425, -0.28575, 0.05715])
        self.assertEqual(mounts['right']['translation_m'], [0.098425, 0.28575, 0.05715])
        self.assertEqual(mounts['forward']['translation_m'], [0.67945, 0.0, -0.10795])
