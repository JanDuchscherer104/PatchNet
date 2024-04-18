from setuptools import find_packages, setup

setup(
    name="dl_solver",
    version="0.1",
    packages=find_packages(),
    package_data={"": ["lib/**/*.py"]},
    install_requires=[
        "pydantic",
        "pydantic-yaml",
        "mlflow",
        "torch",
        "torchvision",
        "torchaudio",
        "pytorch-lightning",
        "torchmetrics",
        "polars",
        "numpy",
        "opencv-python",
    ],
)
