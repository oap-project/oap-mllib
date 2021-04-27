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
echo SPARK_VER=$SPARK_VER
echo GCC Version: $(gcc -dumpversion)
echo =============================

cd $GITHUB_WORKSPACE/mllib-dal

# Build test without profile
$GITHUB_WORKSPACE/dev/ci-build.sh
# Enable signal chaining support for JNI
# export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so

# -Dtest=none to turn off the Java tests

# Test all
# mvn -Dtest=none -Dmaven.test.skip=false test

# Individual test
mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test
mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test
# mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test


SupportedSparkVersions=("spark-3.0.1" "spark-3.0.2" "spark-3.1.1")
# Print array values in  lines
for SPARK_VER in ${SupportedSparkVersions[*]}; do
    echo ""
    echo "========================================"
    echo $SPARK_VER
    echo "========================================"
    # Build test with profile
    SPARK_VER=$SPARK_VER && $GITHUB_WORKSPACE/dev/ci-build.sh

    mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test -P$SPARK_VER
    mvn --no-transfer-progress -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test -P$SPARK_VER
    # mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test -P$SPARK_VER
done

# Yarn cluster test
$GITHUB_WORKSPACE/dev/test-cluster/ci-test-cluster.sh