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


if __name__ == '__main__':
    unittest.main()
