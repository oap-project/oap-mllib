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

versionArray=(
  spark-3.0.0 \
  spark-3.0.1 \
  spark-3.0.2 \
  spark-3.1.1
)

suiteArray=(
  "clustering.MLlibKMeansSuite" \
  "feature.MLlibPCASuite" \
  "recommendation.MLlibALSSuite" \
  "classification.MLlibNaiveBayesSuite" \
  "regression.MLlibLinearRegressionSuite"
)

# Set default version
SPARK_VER=spark-3.0.0

print_usage() {
  echo
  echo Usage: ./test.sh [-p spark-x.x.x] [test suite name]
  echo
  echo Supported Spark versions:
  for version in ${versionArray[*]}
  do
    echo "    $version"
  done
  echo
  echo Supported Test suites:
  for suite in ${suiteArray[*]}
  do
    echo "    $suite"
  done
  echo  
}

while getopts "hp:" opt
do
case $opt in
  p) SPARK_VER=$OPTARG ;;
  h | *)
     print_usage
     exit 1
     ;;
esac
done

shift "$((OPTIND-1))"

print_usage

echo === Testing Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo Spark Version: $SPARK_VER
echo ============================

ALGO=$1

if [[ ! ${versionArray[*]} =~ $SPARK_VER ]]; then
  echo Error: $SPARK_VER version is not supported!
  exit 1
fi

if [[ ! ${suiteArray[*]} =~ $ALGO ]]; then
  echo Error: $ALGO test suite is not supported!
  exit 1
fi

if [[ -z $ALGO ]]; then
  echo
  echo Testing ALL suites...
  echo
  mvn -P$SPARK_VER -Dtest=none test
else
  echo
  echo Testing org.apache.spark.ml.$ALGO ...
  echo
  mvn -P$SPARK_VER -Dtest=none -DwildcardSuites=org.apache.spark.ml.$ALGO test
fi
