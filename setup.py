from __future__ import annotations

from os import path

from setuptools import find_packages, setup

with open(
    path.join(path.abspath(path.dirname(__file__)), "langsuite", "version.py"),
    encoding="utf-8",
) as f:
    exec(f.read())

with open(
    path.join(path.abspath(path.dirname(__file__)), "requirements.txt"),
    encoding="utf-8",
) as f:
    requirements = f.readlines()

setup(
    name="langsuite",
    version=__version__,
    author="BIGAI NLCo Group",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={"console_scripts": ["langsuite=langsuite.__main__:main"]},
    python_requires=">=3.8",
)
