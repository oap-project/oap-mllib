#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Setup Spark envs
source $GITHUB_WORKSPACE/dev/test-cluster/setup-spark-envs.sh

# Setup OAP MLlib envs
cp $GITHUB_WORKSPACE/dev/test-cluster/env.sh $GITHUB_WORKSPACE/conf

cd $GITHUB_WORKSPACE/examples

HOST_NAME=$(hostname -f)
export HDFS_ROOT=hdfs://$HOST_NAME:8020

# Copy examples data to HDFS
hadoop fs -copyFromLocal data /
hadoop fs -find /

# Build and run all examples
./build-all.sh
./run-all-scala.sh
./run-all-pyspark.sh
