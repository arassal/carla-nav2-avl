from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

import yaml

from zedx_vision_costmap import __version__
from zedx_vision_costmap.core import LAYER_NAMES


ROOT = Path(__file__).parents[1]


class ProjectContractTests(unittest.TestCase):
    def test_package_identity_is_consistent(self):
        expected = 'zedx_vision_costmap'
        package_name = ET.parse(ROOT / 'package.xml').getroot().findtext('name')
        setup_text = (ROOT / 'setup.py').read_text()
        self.assertEqual(package_name, expected)
        self.assertIn(f"package_name = '{expected}'", setup_text)
        self.assertTrue((ROOT / 'resource' / expected).is_file())
        self.assertTrue((ROOT / expected / '__init__.py').is_file())
        self.assertTrue((ROOT / 'config' / f'{expected}.yaml').is_file())
        self.assertTrue((ROOT / 'config' / f'{expected}.rviz').is_file())

    def test_legacy_brand_is_absent(self):
        legacy_prefix = 'se' + 'ven'
        legacy_terms = (
            legacy_prefix + '_layer',
            legacy_prefix + '-layer',
            legacy_prefix + ' layer',
        )
        repository = ROOT.parent
        scanned = [repository / 'README.md']
        scanned.extend(path for path in ROOT.rglob('*')
                       if path.is_file() and '__pycache__' not in path.parts)
        scanned.extend(path for path in (repository / '.github').rglob('*') if path.is_file())
        for path in scanned:
            relative = str(path.relative_to(repository)).lower()
            content = path.read_text(errors='ignore').lower()
            for legacy in legacy_terms:
                self.assertNotIn(legacy, relative, f'legacy path: {relative}')
                self.assertNotIn(legacy, content, f'legacy text in: {relative}')

    def test_versions_match(self):
        package_version = ET.parse(ROOT / 'package.xml').getroot().findtext('version')
        setup_text = (ROOT / 'setup.py').read_text()
        setup_version = re.search(r"version='([^']+)'", setup_text).group(1)
        self.assertEqual(__version__, package_version)
        self.assertEqual(__version__, setup_version)

    def test_every_layer_has_configured_weight(self):
        config = yaml.safe_load((ROOT / 'config' / 'zedx_vision_costmap.yaml').read_text())
        weights = config['zedx_vision_costmap']['ros__parameters']['weights']
        self.assertEqual(set(weights), set(LAYER_NAMES))

    def test_rviz_lists_master_and_every_layer(self):
        rviz = (ROOT / 'config' / 'zedx_vision_costmap.rviz').read_text()
        self.assertIn('/zedx_vision_costmap/costmap', rviz)
        for layer in LAYER_NAMES:
            self.assertIn(f'/zedx_vision_costmap/layers/{layer}', rviz)
        self.assertIn('/zedx_vision_costmap/bev/occupancy', rviz)
        self.assertIn('/zedx_vision_costmap/points/fused', rviz)
        for camera in ('zed_front', 'zed_left', 'zed_right'):
            self.assertIn(f'/{camera}/{camera}_node/rgb/color/rect/image', rviz)

    def test_operational_scripts_are_strict_shell_scripts(self):
        for script in (ROOT / 'scripts').glob('*.sh'):
            text = script.read_text()
            self.assertTrue(text.startswith('#!/usr/bin/env bash'))
            self.assertIn('set -', text)

    def test_three_svo_launcher_is_installed_for_ros2_run(self):
        setup_text = (ROOT / 'setup.py').read_text()
        self.assertIn("scripts=['scripts/run_three_svo_vision.sh']", setup_text)

    def test_diagnostic_collection_does_not_dump_all_environment_variables(self):
        text = (ROOT / 'scripts' / 'collect_diagnostics.sh').read_text()
        self.assertNotIn('printenv', text)

    def test_nodes_do_not_overwrite_rclpy_reserved_collections(self):
        reserved = ('_publishers', '_subscriptions', '_timers', '_clients', '_services')
        for path in (ROOT / 'zedx_vision_costmap').glob('*_node.py'):
            text = path.read_text()
            for name in reserved:
                self.assertNotIn(f'self.{name} =', text, f'{path.name} overwrites {name}')

    def test_fusion_rate_matches_perception_rate(self):
        config = yaml.safe_load((ROOT / 'config' / 'zedx_vision_costmap.yaml').read_text())
        fusion_hz = config['zedx_vision_costmap']['ros__parameters']['publish_frequency']
        perception_hz = config['three_zed_perception']['ros__parameters']['processing_frequency']
        self.assertEqual(fusion_hz, perception_hz)

    def test_runtime_contract_is_camera_only(self):
        checked = [
            ROOT / 'zedx_vision_costmap' / 'perception_node.py',
            ROOT / 'zedx_vision_costmap' / 'perception.py',
            ROOT / 'launch' / 'three_svo_costmap.launch.py',
            ROOT / 'config' / 'zedx_vision_costmap.yaml',
        ]
        runtime = '\n'.join(path.read_text().lower() for path in checked)
        for forbidden in ('laserscan', 'velodyne', 'lidar_topic'):
            self.assertNotIn(forbidden, runtime)
        self.assertNotIn('create_subscription(PointCloud2', runtime)

        zed = yaml.safe_load((ROOT / 'config' / 'zed_svo_override.yaml').read_text())
        self.assertNotIn('point_cloud_freq',
                         zed['/**']['ros__parameters'].get('depth', {}))
        mounts = yaml.safe_load((ROOT / 'config' / 'camera_mounts.yaml').read_text())
        self.assertEqual(set(mounts['camera_mounts']), {'left', 'right', 'forward'})

    def test_removed_heuristic_layers_are_absent_from_runtime_contract(self):
        self.assertNotIn('traffic_regulation', LAYER_NAMES)
        self.assertNotIn('road_condition', LAYER_NAMES)
        runtime_files = [
            ROOT / 'setup.py',
            ROOT / 'config' / 'zedx_vision_costmap.yaml',
            ROOT / 'config' / 'zedx_vision_costmap.rviz',
            ROOT / 'launch' / 'three_svo_costmap.launch.py',
            ROOT / 'zedx_vision_costmap' / 'perception.py',
        ]
        runtime = '\n'.join(path.read_text().lower() for path in runtime_files)
        self.assertNotIn('traffic_regulation', runtime)
        self.assertNotIn('road_condition', runtime)
        self.assertFalse((ROOT / 'zedx_vision_costmap' / 'road_condition_node.py').exists())

    def test_default_pipeline_is_vision_only_and_uses_full_mount_rotation(self):
        config = yaml.safe_load((ROOT / 'config' / 'zedx_vision_costmap.yaml').read_text())
        perception = config['three_zed_perception']['ros__parameters']
        self.assertFalse(perception['use_motion_compensation'])
        self.assertFalse(perception['enable_temporal_memory'])
        self.assertFalse(perception['enable_prediction'])
        for name in ('front', 'left', 'right'):
            self.assertEqual(len(perception['mounts'][name]['rpy']), 3)

    def test_quality_and_realtime_svo_profiles_are_distinct(self):
        quality = yaml.safe_load((ROOT / 'config' / 'zed_svo_override.yaml').read_text())
        realtime = yaml.safe_load(
            (ROOT / 'config' / 'zed_svo_realtime_override.yaml').read_text())
        quality_params = quality['/**']['ros__parameters']
        realtime_params = realtime['/**']['ros__parameters']
        self.assertFalse(quality_params['svo']['svo_realtime'])
        self.assertEqual(quality_params['depth']['depth_mode'], 'NEURAL')
        self.assertTrue(realtime_params['svo']['svo_realtime'])
        self.assertEqual(realtime_params['depth']['depth_mode'], 'NEURAL_LIGHT')

    def test_blind_spot_costs_are_ordered_and_nonlethal(self):
        config = yaml.safe_load((ROOT / 'config' / 'zedx_vision_costmap.yaml').read_text())
        perception = config['three_zed_perception']['ros__parameters']
        self.assertLessEqual(perception['blind_spot_clear_cost'],
                             perception['blind_spot_unknown_cost'])
        self.assertLess(perception['blind_spot_unknown_cost'], 100)
        self.assertEqual(len(perception['blind_spot_centers_deg']), 2)

    def test_real_and_synthetic_zed_topics_match(self):
        perception = (ROOT / 'zedx_vision_costmap' / 'perception_node.py').read_text()
        synthetic = (ROOT / 'zedx_vision_costmap' / 'synthetic_zed_node.py').read_text()
        for topic in ('/rgb/color/rect/image', '/rgb/color/rect/camera_info',
                      '/depth/depth_registered'):
            self.assertIn(topic, perception)
            self.assertIn(topic, synthetic)


if __name__ == '__main__':
    unittest.main()
