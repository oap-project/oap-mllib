#!/usr/bin/env bash

# Build and run all examples
cp $GITHUB_WORKSPACE/dev/test-cluster/env.sh $GITHUB_WORKSPACE/conf

cd $GITHUB_WORKSPACE/examples

hadoop fs -mkdir -p /user/$USER
hadoop fs -copyFromLocal data
hadoop fs -ls data

./build-all.sh
./run-all-scala.sh
./run-all-pyspark.sh
