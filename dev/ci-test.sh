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


SupportedSparkVersions=("spark-3.0.0" "spark-3.0.1" "spark-3.0.2" "spark-3.1.1")

for SparkVer in ${SupportedSparkVersions[*]}; do
    echo ""
    echo "========================================"
    echo "Profile: $SparkVer"
    echo "========================================"

    cd $GITHUB_WORKSPACE/mllib-dal
    # Build test with profile
    $GITHUB_WORKSPACE/dev/ci-build.sh $SparkVer

    mvn --no-transfer-progress -P$SparkVer -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test
    mvn --no-transfer-progress -P$SparkVer -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test
    # mvn --no-transfer-progress -P$SparkVer -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test
done

# Yarn cluster test without profile
$GITHUB_WORKSPACE/dev/ci-build.sh
$GITHUB_WORKSPACE/dev/test-cluster/ci-test-cluster.sh
