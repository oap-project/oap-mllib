import os
import sys

from setuptools import setup

setup(
name="oap-mllib",
version=VERSION,
description="",
author="",
author_email="",
url="",
packages=[
    "oap-mllib",
],
include_package_data=True,
package_dir={
    "pyspark.jars": "deps/jars",
    "pyspark.bin": "deps/bin",
    "pyspark.sbin": "deps/sbin",
    "pyspark.python.lib": "lib",
    "pyspark.data": "deps/data",
    "pyspark.licenses": "deps/licenses",
    "pyspark.examples.src.main.python": "deps/examples",
},
package_data={
    "pyspark.jars": ["*.jar"],
    "pyspark.bin": ["*"],
    "pyspark.sbin": [
        "spark-config.sh",
        "spark-daemon.sh",
        "start-history-server.sh",
        "stop-history-server.sh",
    ],
    "pyspark.python.lib": ["*.zip"],
    "pyspark.data": ["*.txt", "*.data"],
    "pyspark.licenses": ["*.txt"],
    "pyspark.examples.src.main.python": ["*.py", "*/*.py"],
},
scripts=scripts,
license="http://www.apache.org/licenses/LICENSE-2.0",
# Don't forget to update python/docs/source/getting_started/install.rst
# if you're updating the versions or dependencies.
install_requires=["py4j==0.10.9.7"],
extras_require={
    "ml": ["numpy>=1.15"],
    "mllib": ["numpy>=1.15"],
    "sql": [
        "pandas>=%s" % _minimum_pandas_version,
        "pyarrow>=%s" % _minimum_pyarrow_version,
        "numpy>=1.15",
    ],
    "pandas_on_spark": [
        "pandas>=%s" % _minimum_pandas_version,
        "pyarrow>=%s" % _minimum_pyarrow_version,
        "numpy>=1.15",
    ],
    "connect": [
        "pandas>=%s" % _minimum_pandas_version,
        "pyarrow>=%s" % _minimum_pyarrow_version,
        "grpcio>=%s" % _minimum_grpc_version,
        "grpcio-status>=%s" % _minimum_grpc_version,
        "googleapis-common-protos>=%s" % _minimum_googleapis_common_protos_version,
        "numpy>=1.15",
    ],
},
python_requires=">=3.7",
classifiers=[
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: Implementation :: CPython",
    "Programming Language :: Python :: Implementation :: PyPy",
    "Typing :: Typed",
],
cmdclass={
    "install": InstallCommand,
},
)
