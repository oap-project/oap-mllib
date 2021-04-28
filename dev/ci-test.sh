#!/usr/bin/env bash

# Setup building envs
source /opt/intel/oneapi/setvars.sh
source /tmp/oneCCL/build/_install/env/setvars.sh

# Check envs for building
if [[ -z $JAVA_HOME ]]; then
 echo JAVA_HOME not defined!
 exit 1
fi

if [[ -z $(which mvn) ]]; then
 echo Maven not found!
 exit 1
fi

if [[ -z $DAALROOT ]]; then
 echo DAALROOT not defined!
 exit 1
fi

if [[ -z $TBBROOT ]]; then
 echo TBBROOT not defined!
 exit 1
fi

if [[ -z $CCL_ROOT ]]; then
 echo CCL_ROOT not defined!
 exit 1
fi

echo === Testing Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo =============================

# cd $GITHUB_WORKSPACE/mllib-dal

# Build test without profile
# $GITHUB_WORKSPACE/dev/ci-build.sh
# Enable signal chaining support for JNI
# export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so

# -Dtest=none to turn off the Java tests

# Test all
# mvn -Dtest=none -Dmaven.test.skip=false test

# Individual test
# mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test
# mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test
# mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test


SupportedSparkVersions=("Spark-3.0.0" "Spark-3.0.1" "Spark-3.0.2" "Spark-3.1.1")
# Print array values in  lines
for SparkVer in ${SupportedSparkVersions[*]}; do
    echo ""
    echo "========================================"
    echo "Profile: $SparkVer"
    echo "========================================"

    cd $GITHUB_WORKSPACE/mllib-dal
    # Build test with profile
    $GITHUB_WORKSPACE/dev/ci-build.sh $SparkVer

    mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test -P$SparkVer
    mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test -P$SparkVer
    # mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test -P$SparkVer
done

# Yarn cluster test
$GITHUB_WORKSPACE/dev/test-cluster/ci-test-cluster.sh