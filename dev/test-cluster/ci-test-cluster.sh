#!/usr/bin/env bash

set -x

# Setup Password-less & Python3
cd $GITHUB_WORKSPACE/dev/test-cluster
./config-ssh.sh
./setup-python3.sh

# Setup Hadoop cluster and envs
source ./setup-cluster.sh

# Build and run all examples
cp $GITHUB_WORKSPACE/dev/test-cluster/env.sh $GITHUB_WORKSPACE/conf

cd $GITHUB_WORKSPACE/examples

hadoop fs -mkdir data
hadoop fs -copyFromLocal -f data
hadoop fs -ls data

./build-all.sh
./run-all-scala.sh
./run-all-pyspark.sh
