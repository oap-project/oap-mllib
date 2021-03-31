#!/usr/bin/env bash

set -x

# Setup Python3 and Spark cluster
cd $GITHUB_WORKSPACE/dev/test-cluster
./config-ssh.sh
# ./setup-cluster.sh
# ./setup-python3-env.sh

# # Build and run all examples
# source $GITHUB_WORKSPACE/dev/test-cluster/env.sh

# cd $GITHUB_WORKSPACE/examples

# hadoop fs -copyFromLocal data
# hadoop fs -ls data

# ./build-all.sh
# ./run-all-scala.sh
# ./run-all-pyspark.sh
