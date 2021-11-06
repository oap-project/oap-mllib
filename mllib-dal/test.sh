#!/usr/bin/env bash

SCRIPT_DIR=$( cd $(dirname ${BASH_SOURCE[0]}) && pwd )
OAP_MLLIB_ROOT=$(cd $SCRIPT_DIR/.. && pwd)
source $OAP_MLLIB_ROOT/RELEASE

if [[ -n $DAALROOT ]]; then
  echo
  echo ====================================================================================
  echo WARNING: DAALROOT detected. It is recommended to test without oneAPI environment!
  echo ====================================================================================
  echo
fi

# Unset FI_PROVIDER_PATH if present otherwise may hang
if [[ -n $FI_PROVIDER_PATH ]]; then
  echo ====================================================================================
  echo WARNING: FI_PROVIDER_PATH detected. Will unset FI_PROVIDER_PATH before proceeding!
  unset FI_PROVIDER_PATH
  echo ====================================================================================
fi

if [[ ! -f target/oap-mllib-$OAP_MLLIB_VERSION.jar ]]; then
  echo Please run ./build.sh first to do a complete build before testing!
  exit 1
fi

# Check envs for building
if [[ -z $JAVA_HOME ]]; then
 echo JAVA_HOME not defined!
 exit 1
fi

if [[ -z $(which mvn) ]]; then
 echo Maven not found!
 exit 1
fi

export OAP_MLLIB_TESTING=true

versionArray=(
  spark-3.1.1 \
  spark-3.1.2 \
  spark-3.2.0
)

suiteArray=(
  "clustering.MLlibKMeansSuite" \
  "feature.MLlibPCASuite" \
  "recommendation.MLlibALSSuite" \
  "classification.MLlibNaiveBayesSuite" \
  "regression.MLlibLinearRegressionSuite" \
  "stat.MLlibCorrelationSuite"
)

# Set default version
SPARK_VER=$SPARK_VERSION
MVN_NO_TRANSFER_PROGRESS=

print_usage() {
  echo
  echo Usage: ./test.sh [-p spark-x.x.x] [-q] [-h] [test suite name]
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

while getopts "hqp:" opt
do
case $opt in
  p) SPARK_VER=$OPTARG ;;
  q) MVN_NO_TRANSFER_PROGRESS=--no-transfer-progress ;;
  h | *)
     print_usage
     exit 1
     ;;
esac
done

shift "$((OPTIND-1))"

print_usage

export PLATFORM_PROFILE=CPU_ONLY_PROFILE

echo === Testing Environments ===
echo JAVA_HOME=$JAVA_HOME
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Spark Version: $SPARK_VER
echo Platform Profile: $PLATFORM_PROFILE
echo ============================

# SUITE=$1

SUITE=feature.MLlibPCASuite

if [[ ! ${versionArray[*]} =~ $SPARK_VER ]]; then
  echo Error: $SPARK_VER version is not supported!
  exit 1
fi

if [[ ! ${suiteArray[*]} =~ $SUITE ]]; then
  echo Error: $SUITE test suite is not supported!
  exit 1
fi

if [[ -z $SUITE ]]; then
  echo
  echo Testing ALL suites...
  echo
  mvn $MVN_NO_TRANSFER_PROGRESS -Dtest=none test
else
  echo
  echo Testing org.apache.spark.ml.$SUITE ...
  echo
  mvn $MVN_NO_TRANSFER_PROGRESS -Dtest=none -DwildcardSuites=org.apache.spark.ml.$SUITE test
fi
