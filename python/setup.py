#!/usr/bin/env python3

import importlib.util
import glob
import os
import sys
import subprocess

from setuptools import setup
from setuptools.command.install import install
from shutil import copyfile, copytree, rmtree

# A temporary path so we can access above the Python project root and fetch scripts and jars we need
TEMP_PATH = "deps"
OAP_HOME = os.path.abspath("../")

# Provide guidance about how to use setup.py
incorrect_invocation_message = """
If you are installing oap_mllib from spark source, you must first build Spark and
run sdist.
    To build Spark with maven you can run:
      ./build/mvn -DskipTests clean package
    Building the source dist is done in the Python directory:
      cd python
      python setup.py sdist
      pip install dist/*.tar.gz"""

# Figure out is the jar compiled.
JAR_PATH = os.path.join(OAP_HOME, "mllib-dal/target")
EXAMPLES_PATH = os.path.join(OAP_HOME, "examples")

JARS_TARGET = os.path.join(TEMP_PATH, "jars")
EXAMPLES_TARGET = os.path.join(TEMP_PATH, "examples")


try:
    copytree(JAR_PATH, JARS_TARGET)
    copytree(EXAMPLES_PATH, EXAMPLES_TARGET)

    with open('../README.md') as f:
        long_description = f.read()

    VERSION = "1.6.0"

    setup(
        name='oap_mllib',
        version=VERSION,
        description='OAP MLlib',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=['oap_mllib',
            'oap_mllib.jars',
            'oap_mllib.examples'],
        install_requires=[],
        package_dir={
            'oap_mllib.jars': 'deps/jars',
            'oap_mllib.examples': 'deps/examples'
        },
        package_data={
            'oap_mllib.jars': ['*.jar'],
            'oap_mllib.examples': ['*'],
        },
        python_requires='>=3.6',
    )
finally:
    rmtree(os.path.join(TEMP_PATH, "jars"))
    rmtree(os.path.join(TEMP_PATH, "examples"))
    os.rmdir(TEMP_PATH)
