#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import platform
import sys
from shutil import rmtree

from setuptools import Command, setup, find_packages

# path of the directory where this file is located
here = os.path.abspath(os.path.dirname(__file__))

# query platform informations, e.g. 'macOS-12.0.1-arm64-arm-64bit'
platform_infos = platform.platform()


# What packages are required for this module to be executed?
REQUIRED = [
    "ConfigSpace>=0.4.20,<=0.6.1",
    "deephyper[default,ray,redis-hiredis]>=0.6.0",
    "numpy",
    "openml",
    "pandas>=0.24.2",
    # "py_experimenter>=1.1,<2",
    "pyyaml",
    "scikit-learn>=0.23.1",
    "scipy>=1.7",
    "tqdm>=4.64.0",
    "xgboost>=1.7.6",
]


# What packages are optional?
EXTRAS = {
    "dev": [
        # Packaging
        "twine",
        # Formatter and Linter
        "black==22.6.0",
    ],
}

# Useful commands to build/upload the wheel to PyPI


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


class TestUploadCommand(Command):
    """Support setup.py testupload."""

    description = "Build and publish the package to test.pypi."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload --repository-url https://test.pypi.org/legacy/ dist/*")

        sys.exit()


class TestInstallCommand(Command):
    """Support setup.py testinstall"""

    description = "Install lcdb from TestPyPI."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        self.status("Downloading the package from Test PyPI and installing it")
        os.system("pip install --index-url https://test.pypi.org/simple/ lcdb")

        sys.exit()


# Where the magic happens:
setup(
    name="lcdb",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    cmdclass={
        "upload": UploadCommand,
        "testupload": TestUploadCommand,
        "testinstall": TestInstallCommand,
    },
)