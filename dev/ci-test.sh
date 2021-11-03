#!/usr/bin/env bash

# Setup building envs
source /opt/intel/oneapi/setvars.sh
source /tmp/oneCCL/build/_install/env/setvars.sh

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-builds-deps.sh

# Test for all versions
SupportedSparkVersions=("spark-3.0.0" "spark-3.0.1" "spark-3.0.2" "spark-3.1.1")
for SparkVer in ${SupportedSparkVersions[*]}; do
    echo
    echo "========================================"
    echo "Testing with Spark Version: $SparkVer"
    echo "========================================"
    echo
    cd $GITHUB_WORKSPACE/mllib-dal
    ./build.sh -q
    ./test.sh -q -p $SparkVer
done

# Yarn cluster test with default spark version
cd $GITHUB_WORKSPACE/mllib-dal
./build.sh -q
$GITHUB_WORKSPACE/dev/test-cluster/ci-test-cluster.sh
