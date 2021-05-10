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
 echo SPARK_VER not defined, using default version spark-3.0.0.
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

# Individual test
if [[ -z $SPARK_VER ]]; then
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite clean test
 mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite clean test
# mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite clean test
else
 mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.IntelKMeansSuite clean test
 mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.IntelPCASuite clean test
# mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.IntelALSSuite clean test
fi
