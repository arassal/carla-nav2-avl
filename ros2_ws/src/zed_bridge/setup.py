from setuptools import setup

package_name = 'zed_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'zed_bridge_node = zed_bridge.bridge_node:main',
            'costmap_node = zed_bridge.costmap_node:main',
        ],
    },
)
