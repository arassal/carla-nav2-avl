from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

import yaml

from seven_layer_costmap import __version__
from seven_layer_costmap.core import LAYER_NAMES


ROOT = Path(__file__).parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_versions_match(self):
        package_version = ET.parse(ROOT / 'package.xml').getroot().findtext('version')
        setup_text = (ROOT / 'setup.py').read_text()
        setup_version = re.search(r"version='([^']+)'", setup_text).group(1)
        self.assertEqual(__version__, package_version)
        self.assertEqual(__version__, setup_version)

    def test_every_layer_has_configured_weight(self):
        config = yaml.safe_load((ROOT / 'config' / 'seven_layer_costmap.yaml').read_text())
        weights = config['seven_layer_costmap']['ros__parameters']['weights']
        self.assertEqual(set(weights), set(LAYER_NAMES))

    def test_rviz_lists_master_and_every_layer(self):
        rviz = (ROOT / 'config' / 'seven_layer_costmap.rviz').read_text()
        self.assertIn('/seven_layer_costmap/costmap', rviz)
        for layer in LAYER_NAMES:
            self.assertIn(f'/seven_layer_costmap/layers/{layer}', rviz)

    def test_operational_scripts_are_strict_shell_scripts(self):
        for script in (ROOT / 'scripts').glob('*.sh'):
            text = script.read_text()
            self.assertTrue(text.startswith('#!/usr/bin/env bash'))
            self.assertIn('set -', text)

    def test_diagnostic_collection_does_not_dump_all_environment_variables(self):
        text = (ROOT / 'scripts' / 'collect_diagnostics.sh').read_text()
        self.assertNotIn('printenv', text)

    def test_timestamp_offsets_match_between_perception_nodes(self):
        config = yaml.safe_load((ROOT / 'config' / 'seven_layer_costmap.yaml').read_text())
        perception = config['three_zed_perception']['ros__parameters']['timestamp_offsets_s']
        road = config['road_condition_layer']['ros__parameters']['timestamp_offsets_s']
        self.assertEqual(perception, road)

    def test_nodes_do_not_overwrite_rclpy_reserved_collections(self):
        reserved = ('_publishers', '_subscriptions', '_timers', '_clients', '_services')
        for path in (ROOT / 'seven_layer_costmap').glob('*_node.py'):
            text = path.read_text()
            for name in reserved:
                self.assertNotIn(f'self.{name} =', text, f'{path.name} overwrites {name}')

    def test_fusion_rate_matches_perception_rate(self):
        config = yaml.safe_load((ROOT / 'config' / 'seven_layer_costmap.yaml').read_text())
        fusion_hz = config['seven_layer_costmap']['ros__parameters']['publish_frequency']
        perception_hz = config['three_zed_perception']['ros__parameters']['processing_frequency']
        self.assertEqual(fusion_hz, perception_hz)

    def test_runtime_contract_is_camera_only(self):
        checked = [
            ROOT / 'seven_layer_costmap' / 'perception_node.py',
            ROOT / 'seven_layer_costmap' / 'perception.py',
            ROOT / 'launch' / 'three_svo_costmap.launch.py',
            ROOT / 'config' / 'seven_layer_costmap.yaml',
        ]
        runtime = '\n'.join(path.read_text().lower() for path in checked)
        for forbidden in ('laserscan', 'pointcloud2', 'velodyne', 'lidar_topic'):
            self.assertNotIn(forbidden, runtime)

        zed = yaml.safe_load((ROOT / 'config' / 'zed_svo_override.yaml').read_text())
        self.assertEqual(zed['/**']['ros__parameters']['depth']['point_cloud_freq'], 0.0)
        mounts = yaml.safe_load((ROOT / 'config' / 'camera_mounts.yaml').read_text())
        self.assertEqual(set(mounts['camera_mounts']), {'left', 'right', 'forward'})

    def test_blind_spot_costs_are_ordered_and_nonlethal(self):
        config = yaml.safe_load((ROOT / 'config' / 'seven_layer_costmap.yaml').read_text())
        perception = config['three_zed_perception']['ros__parameters']
        self.assertLessEqual(perception['blind_spot_clear_cost'],
                             perception['blind_spot_unknown_cost'])
        self.assertLess(perception['blind_spot_unknown_cost'], 100)
        self.assertEqual(len(perception['blind_spot_centers_deg']), 2)


if __name__ == '__main__':
    unittest.main()
