#!/usr/bin/env bash

cd $GITHUB_WORKSPACE

source ./env.sh

cd examples

hadoop fs -copyFromLocal data
hadoop fs -ls data

./build.sh
./run-all-scala.sh
./run-all-pyspark.sh
