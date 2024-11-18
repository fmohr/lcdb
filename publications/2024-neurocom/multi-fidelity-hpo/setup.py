from setuptools import setup

REQUIRED = ["matplotlib", "pyaml", "optuna"]


setup(name="dhexp", packages=["dhexp"], install_requires=REQUIRED)
