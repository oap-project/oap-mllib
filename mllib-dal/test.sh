#!/usr/bin/env bash

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

if [[ -z $1 ]]; then
 echo SPARK_VER not defined, using default (3.0.0).
else
 SPARK_VER=$1
fi

echo === Testing Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo SPARK_VER=$SPARK_VER
echo =============================

# Enable signal chaining support for JNI
# export LD_PRELOAD=$JAVA_HOME/jre/lib/amd64/libjsig.so

# -Dtest=none to turn off the Java tests

# Test all
# mvn -Dtest=none -Dmaven.test.skip=false test

# Individual test
if [[ -z $SPARK_VER ]]; then
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test
# mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test
else
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite test -P$SPARK_VER
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite test -P$SPARK_VER
# mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite test -P$SPARK_VER
fi
