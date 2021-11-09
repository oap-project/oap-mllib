#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup building envs
source /opt/intel/oneapi/setvars.sh

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-build-deps.sh

# Test for all versions
SupportedSparkVersions=("spark-3.1.1" "spark-3.1.2" "spark-3.2.0")
for SparkVer in ${SupportedSparkVersions[*]}; do
    echo
    echo "========================================"
    echo "Testing with Spark Version: $SparkVer"
    echo "========================================"
    echo
    cd $GITHUB_WORKSPACE/mllib-dal
    ./build.sh -q
    unset LD_LIBRARY_PATH
    ./test.sh -q -p $SparkVer
done
