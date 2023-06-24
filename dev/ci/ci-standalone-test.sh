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
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-build-deps.sh
./build.sh -p CPU_GPU_PROFILE -q

# Setup cluster
source $GITHUB_WORKSPACE/dev/test-cluster/standalone/setup-cluster.sh

# Setup OAP MLlib envs
cp $GITHUB_WORKSPACE/dev/test-cluster/standalone/env.sh $GITHUB_WORKSPACE/conf
cd $GITHUB_WORKSPACE/examples


echo "========================================="
echo "Cluster Testing with Spark Version: $SPARK_VERSION"
echo "========================================="

# Build and run all examples
./build-all-scala.sh
./run-all-scala-cpu.sh
./run-all-pyspark.sh
