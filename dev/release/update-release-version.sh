#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

[[ -f $SCRIPT_DIR/../RELEASE ]] && source $SCRIPT_DIR/../RELEASE

if [[ -z $OAP_MLLIB_VERSION ]]; then
  echo OAP_MLLIB_VERSION not defined in RELEASE!
  exit 1
fi

if [[ -z $SPARK_VERSION ]]; then
  echo SPARK_VERSION not defined in RELEASE!
  exit 1
fi

echo Updating release version to OAP MLlib $OAP_MLLIB_VERSION for Spark $SPARK_VERSION ...

cd $SCRIPT_DIR/../mllib-dal
mvn versions:set -DnewVersion=$OAP_MLLIB_VERSION
mvn versions:set-property -Dproperty=spark.version -DnewVersion=$SPARK_VERSION

exampleDirs=(kmeans pca als naive-bayes linear-regression correlation)

cd $SCRIPT_DIR/../examples

for dir in ${exampleDirs[*]}
do
  cd $dir
  mvn versions:set -DnewVersion=$OAP_MLLIB_VERSION
  mvn versions:set-property -Dproperty=spark.version -DnewVersion=$SPARK_VERSION
  cd ..  
done
