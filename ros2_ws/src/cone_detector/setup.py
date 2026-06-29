from setuptools import find_packages, setup

package_name = 'cone_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jchy',
    maintainer_email='jchy@cpp.edu',
    description='ROS 2 cone detector using OpenCV',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cone_detector = cone_detector.cone_detector_node:main',
        ],
    },
)