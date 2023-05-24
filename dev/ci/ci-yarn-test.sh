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
./build.sh -p CPU_ONLY_PROFILE -q

# Setup cluster
source $GITHUB_WORKSPACE/dev/test-cluster/yarn/setup-cluster.sh

# Setup OAP MLlib envs
cp $GITHUB_WORKSPACE/dev/test-cluster/yarn/env.sh $GITHUB_WORKSPACE/conf
cd $GITHUB_WORKSPACE/examples

# Copy examples data to HDFS
hadoop fs -copyFromLocal data /
hadoop fs -find /

echo "========================================="
echo "Cluster Testing with Spark Version: $SPARK_VERSION"
echo "========================================="

# Build and run all examplesdebug#./build-all-scala.sh
#./run-all-scala.sh
#./run-all-pyspark.sh
