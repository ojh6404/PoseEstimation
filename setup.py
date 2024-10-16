import os
from setuptools import setup, find_packages


def _post_install():
    CACHE_DIR = os.path.expanduser("~/.cache/pose_estimation")
    os.makedirs(CACHE_DIR, exist_ok=True)


setup(
    name="pose_estimation",
    packages=find_packages(),
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="Pose Estimation Package for robotics applications.",
    author="Jihoon Oh",
    url="https://github.com/ojh6404/PoseEstimation",
    author_email="ojh6404@gmail.com",
    version="0.0.1",
)

_post_install()
