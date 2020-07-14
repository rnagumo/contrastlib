
from setuptools import setup, find_packages


install_requires = [
    "torch==1.5.1",
    "torchvision==0.6.1",
    "scikit-learn==0.23.1",
]


setup(
    name="contrastlib",
    version="0.1",
    description="Contrastive learning by PyTorch",
    packages=find_packages(),
    install_requires=install_requires,
)
