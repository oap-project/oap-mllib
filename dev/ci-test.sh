#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Setup building envs
source /opt/intel/oneapi/setvars.sh

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-build-deps.sh

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
