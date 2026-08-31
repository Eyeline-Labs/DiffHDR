import os
from setuptools import setup, find_packages

setup(
    name="diffsynth",
    version="1.0.0",
    description="DiffHDR: HDR video reconstruction from LDR using diffusion models",
    packages=find_packages(),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
