import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="sonata",
    py_modules=["sonata"],
    version="1.0",
    description="",
    author="Xiaoyang Wu",
    packages=find_packages(exclude=["demo*"]),
    include_package_data=True,
)
