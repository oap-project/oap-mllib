#!/usr/bin/env bash

# Setup Spark envs
source $GITHUB_WORKSPACE/dev/test-cluster/setup-spark-envs.sh

# Setup OAP MLlib envs
cp $GITHUB_WORKSPACE/dev/test-cluster/env.sh $GITHUB_WORKSPACE/conf

cd $GITHUB_WORKSPACE/examples

# Copy examples data to HDFS
hadoop fs -mkdir -p /user/$USER
hadoop fs -copyFromLocal data
hadoop fs -ls data

# Build and run all examples
./build-all.sh
./run-all-scala.sh
./run-all-pyspark.sh
