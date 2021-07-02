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

# Clean
mvn clean

# Run individual tests
if [[ -z $SPARK_VER ]]; then
  mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.MLlibKMeansSuite test
  mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.MLlibPCASuite test
  mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.MLlibALSSuite test
  mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.classification.MLlibNaiveBayesSuite test
  mvn -Dtest=none -DwildcardSuites=org.apache.spark.ml.regression.MLlibLinearRegressionSuite test
else
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.clustering.MLlibKMeansSuite test
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.feature.MLlibPCASuite test
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.recommendation.MLlibALSSuite test
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.classification.MLlibNaiveBayesSuite test
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.regression.MLlibLinearRegressionSuite test
fi
