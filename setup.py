import os
from setuptools import setup

name = "blobmodel"

with open("README.md") as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=name,
    description="2d model of propagating blobs with random parameters. ",
    url="https://github.com/uit-cosmo/2d_propagating_blobs/blob/main/",
    author="Gregor Decristoforo",
    author_email="gregor.decristoforo@uit.no",
    license="GPL",
    version="1.0",
    packages=["blobmodel"],
    python_requires=">=3.9",
    install_requires=[
        "xarray>=0.11",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.3",
        "tqdm>=4.62.2",
        "pytest>=3.3.0",
        "nptyping>=1.4.4",
    ],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    zip_safe=False,
)
