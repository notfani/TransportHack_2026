from glob import glob
from setuptools import find_packages, setup

setup(
    name="tram_odometry", version="0.2.0",
    packages=find_packages(exclude=("tests",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/tram_odometry"]),
        ("share/tram_odometry", ["package.xml", "runtime_manifest.json"]),
        ("share/tram_odometry/config", glob("config/*.yaml")+glob("config/*.json")),
        ("share/tram_odometry/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    python_requires=">=3.10",
    zip_safe=True,
    maintainer="Tram hackathon team", maintainer_email="team@example.invalid",
    description="Reference causal wheel odometry adapter for ROS 2 Humble",
    license="Apache-2.0",
    entry_points={"console_scripts": ["odometry_node = tram_odometry.node:main"]},
)
