from setuptools import find_packages, setup

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='merabro',
    maintainer_email='53960016+prakash-aryan@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller_node = controller.controller_node:main',
            'sdc_demo_node = controller.sdc_demo_node:main',
            'sdc_viz_node = controller.sdc_viz_node:main',
            'sdc_rerun_node = controller.sdc_rerun_node:main',
            'carla_localization = controller.carla_localization:main',
            'lane_path_node = controller.lane_path_node:main',
            'cmd_to_carla = controller.cmd_to_carla:main',
        ],
    },
)
