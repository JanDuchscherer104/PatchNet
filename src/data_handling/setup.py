from setuptools import find_packages, setup

setup(
    name="data_handling",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pydantic",
        "pydantic-yaml",
        "polars",
        "opencv-python",
        "numpy",
        "swifter",
    ],
)
