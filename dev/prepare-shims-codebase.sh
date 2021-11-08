#!/usr/bin/env bash

# set -x

if [[ ! $# -eq 1 ]]; then
    echo
    echo "Usage: ./prepare-shim-codebash.sh <spark-source-code-dir>"
    echo
    exit 0
fi

SPARK_VERSIONS=(
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
    org/apache/spark/ml/optim/aggregator/HuberAggregator.scala
    org/apache/spark/ml/optim/aggregator/LeastSquaresAggregator.scala
    org/apache/spark/mllib/clustering/KMeans.scala
)

SCRIPT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

SPARK_SOURCE_DIR=$1
SOURCE_DIR=$SPARK_SOURCE_DIR/mllib/src/main/scala
TARGET_DIR=$SCRIPT_DIR/shims-sources

copy_sources() {
    VERSION=$1
    echo ==========================
    echo Checking out v$VERSION ...
    git checkout v$VERSION
    git status
    echo
    echo Copying sources for v$VERSION ...
    for file in ${FILES[*]}; do
        rsync -av --relative $SOURCE_DIR/./$file $TARGET_DIR/$VERSION
    done
    echo
}

[ -d $TARGET_DIR ] || mkdir $TARGET_DIR

cd $SPARK_SOURCE_DIR

for VERSION in ${SPARK_VERSIONS[*]}; do
    mkdir -p $TARGET_DIR/$VERSION
    copy_sources $VERSION
done

[[ -n $(which tree) ]] && tree $TARGET_DIR
