##### \* LEGAL NOTICE: Your use of this software and any required dependent software (the "Software Package") is subject to the terms and conditions of the software license agreements for the Software Package, which may also include notices, disclaimers, or license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the "TPP.txt" or other similarly-named text file included with the Software Package for additional details.

##### \* Optimized Analytics Package for Spark* Platform is under Apache 2.0 (https://www.apache.org/licenses/LICENSE-2.0).

# OAP MLlib

## Overview

OAP MLlib is an optimized package to accelerate machine learning algorithms in  [Apache Spark MLlib](https://spark.apache.org/mllib).  It is compatible with Spark MLlib and leverages open source [Intel® oneAPI Data Analytics Library (oneDAL)](https://github.com/oneapi-src/oneDAL) to provide highly optimized algorithms and get most out of CPU and GPU capabilities. It also take advantage of open source [Intel® oneAPI Collective Communications Library (oneCCL)](https://github.com/oneapi-src/oneCCL) to provide efficient communication patterns in multi-node multi-GPU clusters.

## Compatibility

OAP MLlib tried to maintain the same API interfaces and produce same results that are identical with Spark MLlib. However due to the nature of float point operations, there may be some small deviation from the original result, we will try our best to make sure the error is within acceptable range.
For those algorithms that are not accelerated by OAP MLlib, the original Spark MLlib one will be used. 

## Online Documentation

You can find the all the OAP MLlib documents on the [project web page](https://oap-project.github.io/oap-mllib).

## Getting Started

### Java/Scala Users Preferred

Use a pre-built OAP MLlib JAR to get started. You can firstly download OAP package from [OAP-JARs-Tarball](https://github.com/oap-project/oap-tools/releases/download/v1.1.1-spark-3.1.1/oap-1.1.1-bin-spark-3.1.1.tar.gz) and extract this Tarball to get `oap-mllib-x.x.x.jar` under `oap-1.1.1-bin-spark-3.1.1/jars`.

Then you can refer to the following [Running](#running) section to try out.

### Python/PySpark Users Preferred

Use a pre-built JAR to get started. If you have finished [OAP-Installation-Guide](./docs/OAP-Installation-Guide.md), you can find compiled OAP MLlib JAR `oap-mllib-x.x.x.jar` in `$HOME/miniconda2/envs/oapenv/oap_jars/`.

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

### Sanity Check

#### Setup `env.sh`
```
    $ cd conf
    $ cp env.sh.template env.sh
```
Edit related variables in "`Minimun Settings`" of `env.sh`

#### Upload example data files to HDFS
```
    $ cd examples
    $ hadoop fs -mkdir -p /user/$USER
    $ hadoop fs -copyFromLocal data
    $ hadoop fs -ls data
```
#### Run K-means

```
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
* Intel® oneAPI Toolkits 2021.2+ Components:
    - DPC++/C++ Compiler (dpcpp/clang++)
    - Data Analytics Library (oneDAL)
    - Threading Building Blocks (oneTBB)
* [Open Source Intel® oneAPI Collective Communications Library (oneCCL)](https://github.com/oneapi-src/oneCCL)

Intel® oneAPI Toolkits and its components can be downloaded and install from [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html). Installation process for oneAPI using Package Managers (YUM (DNF), APT, and ZYPPER) is also available. Generally you only need to install oneAPI Base Toolkit for Linux with all or selected components mentioned above. Instead of using oneCCL included in Intel® oneAPI Toolkits, we prefer to build from open source oneCCL to resolve some bugs.

More details about oneAPI can be found [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi.html).

Scala and Java dependency descriptions are already included in Maven POM file. 

***Note:*** You can refer to [this script](dev/install-build-deps-centos.sh) to install correct dependencies: DPC++/C++, oneDAL, oneTBB, oneCCL.

### Build

####  Building oneCCL

To clone and build from open source oneCCL, run the following commands:
```
	$ git clone https://github.com/oneapi-src/oneCCL
        $ cd oneCCL
        $ git checkout 2021.2
	$ mkdir build && cd build
	$ cmake ..
	$ make -j install
```

The generated files will be placed in `/your/oneCCL_source_code/build/_install`

#### Building OAP MLlib

To clone and checkout source code, run the following commands:
```
    $ git clone https://github.com/oap-project/oap-mllib.git   
```
__Optional__ to checkout specific release branch:
```
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

```
	$ source /opt/intel/oneapi/setvars.sh
	$ source /your/oneCCL_source_code/build/_install/env/setvars.sh
```

__Be noticed we are using our own built oneCCL instead, we should source oneCCL's `setvars.sh` to overwrite oneAPI one.__

You can also refer to [this CI script](dev/ci-build.sh) to setup the building environments.

If you prefer to buid your own open source [oneDAL](https://github.com/oneapi-src/oneDAL), [oneTBB](https://github.com/oneapi-src/oneTBB) versions rather than use the ones included in oneAPI TookKits, you can refer to the related build instructions and manually source `setvars.sh` accordingly.

To build, run the following commands: 
```
    $ cd mllib-dal
    $ ./build.sh
```

The target can be built against different Spark versions by specifying profile with <spark-x.x.x>. E.g.
```
    $ ./build.sh spark-3.1.1
```
If no profile parameter is given, the Spark version 3.0.0 will be activated by default.
The built JAR package will be placed in `target` directory with the name `oap-mllib-x.x.x.jar`.

## Examples

Example         |  Description
----------------|---------------------------
kmeans          |  K-means example for Scala
kmeans-pyspark  |  K-means example for PySpark
pca             |  PCA example for Scala
pca-pyspark     |  PCA example for PySpark
als             |  ALS example for Scala
als-pyspark     |  ALS example for PySpark

## List of Accelerated Algorithms

Algorithm | Category | Maturity
----------|----------|-------------
K-Means   | CPU      | Experimental
PCA       | CPU      | Experimental
ALS       | CPU      | Experimental
