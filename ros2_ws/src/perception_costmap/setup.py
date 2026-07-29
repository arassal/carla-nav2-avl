import os
from glob import glob
from setuptools import setup

package_name = 'perception_costmap'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexander Assal',
    maintainer_email='alexander@assalfamily.com',
    description='Camera + lidar perception publishing a Nav2-compatible costmap.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'costmap_node = perception_costmap.costmap_node:main',
        ],
    },
)
