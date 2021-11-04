##### \* LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package") is subject to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the "TPP.txt" or other similarly-named text file included with the Software Package for additional details.

##### \* Optimized Analytics Package for Spark* Platform is under Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

# OAP MLlib

## Overview

OAP MLlib is an optimized package to accelerate machine learning algorithms in  [Apache Spark MLlib](https://spark.apache.org/mllib).  It is compatible with Spark MLlib and leverages open source [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL) to provide highly optimized algorithms and get most out of CPU and GPU capabilities. It also take advantage of open source [Intel® oneAPI Collective Communications Library (oneCCL)](https://github.com/oneapi-src/oneCCL) to provide efficient communication patterns in multi-node multi-GPU clusters.

## Compatibility

OAP MLlib maintains the same API interfaces with Spark MLlib. That means the application built with Spark MLlib can be running directly with minimum configuration.

Most of the algorithms can produce the same results that are identical with Spark MLlib. However due to the nature of distributed float point operations, there may be some small deviation from the original result, we will make sure the error is within acceptable range and the accuracy is on par with Spark MLlib.

For those algorithms that are not accelerated by OAP MLlib, the original Spark MLlib one will be used.

## Online Documentation

You can find the all the OAP MLlib documents on the [project web page](https://oap-project.github.io/oap-mllib).

## Getting Started

### Java/Scala Users Preferred

Use a pre-built OAP MLlib JAR to get started. You can firstly download OAP package from [OAP-JARs-Tarball](https://github.com/oap-project/oap-tools/releases/download/v1.2.0/oap-1.2.0-bin.tar.gz) and extract this Tarball to get `oap-mllib-x.x.x.jar` under `oap-1.2.0-bin-spark-3.1.1/jars`.

Then you can refer to the following [Running](#running) section to try out.

### Python/PySpark Users Preferred

Use a pre-built JAR to get started. If you have finished [OAP Installation Guide](./docs/OAP-Installation-Guide.md), you can find compiled OAP MLlib JAR `oap-mllib-x.x.x.jar` in `$HOME/miniconda2/envs/oapenv/oap_jars/`.

Then you can refer to the following [Running](#running) section to try out.

### Building From Scratch

You can also build the package from source code, please refer to [Building](#building) section.

## Running

### Prerequisites

* CentOS 7.0+, Ubuntu 18.04 LTS+
* Java JRE 8.0+ Runtime
* Apache Spark 3.0.0+

Generally, our common system requirements are the same with Intel® oneAPI Toolkit, please refer to [here](https://software.intel.com/content/www/us/en/develop/articles/intel-oneapi-base-toolkit-system-requirements.html) for details.

Intel® oneAPI Toolkits components used by the project are already included into JAR package mentioned above. There are no extra installations for cluster nodes.

### Spark Configuration

#### General Configuration

##### YARN Cluster Manager
Users usually run Spark application on __YARN__ with __client__ mode. In that case, you only need to add the following configurations in `spark-defaults.conf` or in `spark-submit` command line before running.

```
# absolute path of the jar for uploading
spark.files                       /path/to/oap-mllib-x.x.x.jar
# absolute path of the jar for driver class path
spark.driver.extraClassPath       /path/to/oap-mllib-x.x.x.jar
# relative path of the jar for executor class path
spark.executor.extraClassPath     ./oap-mllib-x.x.x.jar
```

##### Standalone Cluster Manager
For standalone cluster manager, you need to upload the jar to every node or use shared network folder and then specify absolute paths for extraClassPath.

```
# absolute path of the jar for driver class path
spark.driver.extraClassPath       /path/to/oap-mllib-x.x.x.jar
# absolute path of the jar for executor class path
spark.executor.extraClassPath     /path/to/oap-mllib-x.x.x.jar
```

#### OAP MLlib Specific Configuration

OAP MLlib adopted oneDAL as implementation backend. oneDAL requires enough native memory allocated for each executor. For large dataset, depending on algorithms, you may need to tune `spark.executor.memoryOverhead` to allocate enough native memory. Setting this value to larger than __dataset size / executor number__ is a good starting point.

OAP MLlib expects 1 executor acts as 1 oneCCL rank for compute. As `spark.shuffle.reduceLocality.enabled` option is `true` by default, when the dataset is not evenly distributed accross executors, this option may result in assigning more than 1 rank to single executor and task failing. The error could be fixed by setting `spark.shuffle.reduceLocality.enabled` to `false`.

### Sanity Check

#### Setup `env.sh`
```bash
    $ cd conf
    $ cp env.sh.template env.sh
```
Edit related variables in "`Minimun Settings`" of `env.sh`

#### Upload example data files to HDFS
```bash
    $ cd examples
    $ hadoop fs -mkdir -p /user/$USER
    $ hadoop fs -copyFromLocal data
    $ hadoop fs -ls data
```
#### Run K-means

```bash
    $ cd examples/kmeans
    $ ./build.sh
    $ ./run.sh
```

### PySpark Support

As PySpark-based applications call their Scala couterparts, they shall be supported out-of-box. Examples can be found in the [Examples](#examples) section.

## Building

### Prerequisites

We use [Apache Maven](https://maven.apache.org/) to manage and build source code.  The following tools and libraries are also needed to build OAP MLlib:

* JDK 8.0+
* Apache Maven 3.6.2+
* GNU GCC 4.8.5+
* Intel® oneAPI Base Toolkit (>=2021.4.0) Components :
    - DPC++/C++ Compiler (dpcpp/clang++)
    - Data Analytics Library (oneDAL)
    - Threading Building Blocks (oneTBB)
    - Collective Communications Library (oneCCL)]

Generally you only need to install __Intel® oneAPI Base Toolkit for Linux__ with all or selected components mentioned above. Intel® oneAPI Base Toolkit can be downloaded and installed from [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html). Installation process for oneAPI using Package Managers (YUM (DNF), APT, and ZYPPER) is also available. More details about oneAPI can be found [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html).

Scala and Java dependency descriptions are already included in Maven POM file.

***Note:*** You can refer to [this script](dev/install-build-deps-centos.sh) to install correct dependencies: DPC++/C++, oneDAL, oneTBB, oneCCL.

### Build

To clone and checkout source code, run the following commands:
```bash
    $ git clone https://github.com/oap-project/oap-mllib.git
```
__Optional__ to checkout specific release branch:
```bash
    $ cd oap-mllib && git checkout ${version}
```

We rely on environment variables to find required toolchains and libraries. Please make sure the following environment variables are set for building:

Environment | Description
------------| -----------
JAVA_HOME   | Path to JDK home directory
DAALROOT    | Path to oneDAL home directory
TBB_ROOT    | Path to oneTBB home directory
CCL_ROOT    | Path to oneCCL home directory

We suggest you to source `setvars.sh` script into current shell to setup building environments as following:

```bash
    $ source /opt/intel/oneapi/setvars.sh
```

You can also refer to [this CI script](dev/ci-test.sh) to setup the building environments.

If you prefer to buid your own open source [oneDAL](https://github.com/oneapi-src/oneDAL), [oneTBB](https://github.com/oneapi-src/oneTBB), [oneCCL](https://github.com/oneapi-src/oneCCL) versions rather than use the ones included in oneAPI Base Toolkit, you can refer to the related build instructions and manually source `setvars.sh` accordingly.

To build, run the following commands:
```bash
    $ cd mllib-dal
    $ ./build.sh
```

If no parameter is given, the Spark version __3.1.1__ will be activated by default. You can also specify a different Spark version with option `-p spark-x.x.x`. For example:
```bash
    $ ./build.sh -p spark-3.0.0
```

The built JAR package will be placed in `target` directory with the name `oap-mllib-x.x.x.jar`.

## Examples

### Scala Examples

Example            |  Description
-------------------|-------------------------------------
kmeans             |  K-means example for Scala
pca                |  PCA example for Scala
als                |  ALS example for Scala
naive-bayes        |  Naive Bayes example for Scala
linear-regression  |  Linear Regression example for Scala
correlation        |  Correlation example for Scala

### Python Examples

Example         |  Description
----------------|---------------------------
kmeans-pyspark  |  K-means example for PySpark
pca-pyspark     |  PCA example for PySpark
als-pyspark     |  ALS example for PySpark

## List of Accelerated Algorithms

Algorithm         | CPU | GPU | Maturity
------------------|-----|-----|---------
K-Means           | X   | X   | Stable
PCA               | X   | X   | Stable
ALS               | X   |     | Experimental
Naive Bayes       | X   |     | Stable
Linear Regression | X   |     | Stable
Correlation       | X   | X   | Experimental
