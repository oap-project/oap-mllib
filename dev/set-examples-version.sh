#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

SPARK_VERSION=3.1.1
OAP_VERSION=1.2.0
OAP_EXAMPLE_VERSION=$OAP_VERSION

exampleDirs=(kmeans pca als naive-bayes linear-regression)

cd $SCRIPT_DIR/../examples

for dir in ${exampleDirs[*]}
do
  cd $dir
  mvn versions:set -DnewVersion=$OAP_EXAMPLE_VERSION
  mvn versions:set-property -Dproperty=spark.version -DnewVersion=$SPARK_VERSION
  cd ..  
done
