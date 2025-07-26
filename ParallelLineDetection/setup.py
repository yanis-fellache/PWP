from setuptools import setup, find_packages

setup(
    name="parallel_line_detection_package",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "parallel-line-detect=parallel_line_detection.main:main",
        ],
    },
    description="A package for detecting and processing lines in images using OpenCV.",
    author="Yanis Fellache",
    author_email="pl1000918@ahschool.com",
    url="https://github.com/yanis-fellache/parrallel_line_detection_package",
)