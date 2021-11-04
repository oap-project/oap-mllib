#!/usr/bin/env bash

# set -x

if [[ ! $# -eq 1 ]]; then
    echo
    echo "Usage: ./prepare-shim-codebash.sh <spark-source-code-dir>"
    echo
    exit 0
fi

VERSIONS=(
    3.0.0
    3.0.1
    3.0.2
    3.0.3
    3.1.1
    3.1.2
    3.2.0
)

FILES=(
    org/apache/spark/ml/classification/NaiveBayes.scala
    org/apache/spark/ml/clustering/KMeans.scala
    org/apache/spark/ml/feature/PCA.scala
    org/apache/spark/ml/recommendation/ALS.scala
    org/apache/spark/ml/regression/LinearRegression.scala
    org/apache/spark/ml/stat/Correlation.scala
)

SCRIPT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

SPARK_SOURCE_DIR=$1
SOURCE_DIR=$SPARK_SOURCE_DIR/mllib/src/main/scala
TARGET_DIR=$SCRIPT_DIR/shims-sources

copy_sources() {
    version=$1
    echo ==========================
    echo Checking out v$version ...
    git checkout v$version
    git status
    echo
    echo Copying sources for v$version ...
    for file in ${FILES[*]}; do
        rsync -av --relative $SOURCE_DIR/./$file $TARGET_DIR/$version
    done
    echo
}

[ -d $TARGET_DIR ] || mkdir $TARGET_DIR

cd $SPARK_SOURCE_DIR

for version in ${VERSIONS[*]}; do
    mkdir -p $TARGET_DIR/$version
    copy_sources $version
done

[[ -n $(which tree) ]] && tree $TARGET_DIR
