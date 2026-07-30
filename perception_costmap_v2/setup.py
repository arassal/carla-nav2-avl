from setuptools import setup, find_packages

package_name = "perception_costmap"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "tools"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", [
            "config/perception_costmap.yaml",
            "config/nav2_costmap_params.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="your_name",
    maintainer_email="you@example.com",
    description="Camera+lidar perception publishing a Nav2-compatible costmap (reimplementation, sim-to-real).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "costmap_node = perception_costmap.costmap_node:main",
        ],
    },
)
