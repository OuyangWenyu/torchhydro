#!/usr/bin/env python

"""
Author: Wenyu Ouyang
Date: 2023-07-31 08:40:43
LastEditTime: 2024-05-31 11:34:36
LastEditors: Wenyu Ouyang
Description: The setup script
FilePath: \torchhydro\setup.py
Copyright (c) 2023-2024 Wenyu Ouyang. All rights reserved.
"""
import io
from os import path as op
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

here = op.abspath(op.dirname(__file__))

# get the dependencies and installs
with io.open(op.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [x.strip().replace("git+", "") for x in all_reqs if "git+" not in x]

requirements = []

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Wenyu Ouyang",
    author_email="wenyuouyang@outlook.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="datasets, samplers, transforms, and pre-trained models for hydrology and water resources",
    entry_points={
        "console_scripts": [
            "torchhydro=torchhydro.cli:main",
        ],
    },
    install_requires=install_requires,
    dependency_links=dependency_links,
    license="BSD license",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="torchhydro",
    name="torchhydro",
    packages=find_packages(include=["torchhydro", "torchhydro.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/OuyangWenyu/torchhydro",
    version='0.0.8',
    zip_safe=False,
)
