from glob import glob
from setuptools import find_packages, setup

package_name = 'seven_layer_costmap'

setup(
    name=package_name,
    version='0.8.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/scripts', glob('scripts/*')),
        ('share/' + package_name + '/docs', glob('docs/*')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=True,
    scripts=['scripts/run_three_svo_vision.sh'],
    entry_points={'console_scripts': [
        'costmap_fusion = seven_layer_costmap.fusion_node:main',
        'three_zed_perception = seven_layer_costmap.perception_node:main',
        'synthetic_zed = seven_layer_costmap.synthetic_zed_node:main',
        'synthetic_layers = seven_layer_costmap.synthetic_layers_node:main',
    ]},
)
