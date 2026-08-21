import os
from glob import glob

from setuptools import setup

package_name = "leap_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="geon",
    maintainer_email="geonlee012@gmail.com",
    description="LEAP Hand v1 Lite webcam teleoperation nodes (tracker / retarget / sim twin / safety bridge)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tracker_node = leap_teleop.tracker_node:main",
            "retarget_node = leap_teleop.retarget_node:main",
            "sim_node = leap_teleop.sim_node:main",
            "hand_bridge_node = leap_teleop.hand_bridge_node:main",
            "fake_hand_node = leap_teleop.fake_hand_node:main",
        ],
    },
)
