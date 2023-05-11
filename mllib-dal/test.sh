#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

SCRIPT_DIR=$( cd $(dirname ${BASH_SOURCE[0]}) && pwd )
OAP_MLLIB_ROOT=$(cd $SCRIPT_DIR/.. && pwd)
source $OAP_MLLIB_ROOT/RELEASE

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

suiteArray=(
)

MVN_NO_TRANSFER_PROGRESS=

print_usage() {
  echo
  echo "Usage: ./test.sh [-p <CPU_ONLY_PROFILE | CPU_GPU_PROFILE>] [-q] [-h] [-d] <test suite name>"
  echo
  echo "-p  Supported Platform Profiles:"
    echo "    CPU_ONLY_PROFILE"
    echo "    CPU_GPU_PROFILE"
  echo
}

while getopts "p:qd:h" opt
do
case $opt in
  p) PLATFORM_OPT=$OPTARG ;;
  q) MVN_NO_TRANSFER_PROGRESS=--no-transfer-progress ;;
  d) DEVICE_OPT=$OPTARG ;;
  h | *)
     print_usage
     exit 1
     ;;
esac
done

shift "$((OPTIND-1))"

SUITE=$*

print_usage

if [[ ! ($PLATFORM_PROFILE == CPU_ONLY_PROFILE || $PLATFORM_PROFILE == CPU_GPU_PROFILE) ]]; then
  echo
  echo Platform Profile should be CPU_ONLY_PROFILE or CPU_GPU_PROFILE, but \"$PLATFORM_PROFILE\" found!
  echo
  exit 1
fi

if [ "HOST" = $DEVICE_OPT ]; then
  suiteArray=(
    "HomogenTableTest" \
    "ColumnAccessorTest" \
    "RowAccessorTest" \
    "com.intel.oap.mllib.ConvertHomogenTableSuite"
  )
elif [ "CPU" = $DEVICE_OPT ]; then
  suiteArray=(
    "org.apache.spark.ml.clustering.MLlibKMeansSuite" \
    "org.apache.spark.ml.feature.MLlibPCASuite" \
    "org.apache.spark.ml.recommendation.MLlibALSSuite" \
    "org.apache.spark.ml.classification.MLlibNaiveBayesSuite" \
    "org.apache.spark.ml.regression.MLlibLinearRegressionSuite" \
    "org.apache.spark.ml.stat.MLlibCorrelationSuite" \
    "org.apache.spark.ml.stat.MLlibSummarizerSuite"
  )
elif [ "GPU" = $DEVICE_OPT ]; then
  suiteArray=(
    "org.apache.spark.ml.regression.MLlibLinearRegressionSuite"
  )
else
  echo Error: $DEVICE_OPT is not supported!
  exit 1
fi

if [[ ! ${suiteArray[*]} =~ $SUITE ]]; then
  echo Error: $SUITE test suite is not supported!
  exit 1
fi

# Import RELEASE envs
SCRIPT_DIR=$( cd $(dirname ${BASH_SOURCE[0]}) && pwd )
OAP_MLLIB_ROOT=$(cd $SCRIPT_DIR/.. && pwd)
source $OAP_MLLIB_ROOT/RELEASE

export SPARK_VERSION=${SPARK_OPT:-$SPARK_VERSION}
export PLATFORM_PROFILE=${PLATFORM_OPT:-$PLATFORM_PROFILE}

echo "computeDevice : $DEVICE_OPT"

echo
echo === Testing Environments ===
echo JAVA_HOME=$JAVA_HOME
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Spark Version: $SPARK_VERSION
echo Platform Profile: $PLATFORM_PROFILE
echo ============================
echo

if [[ -z $SUITE ]]; then
  for suite in ${suiteArray[*]}
  do
    if [[ $suite == *"com.intel.oap.mllib"* ]]; then
      echo
      echo Testing $suite ...
      echo
      mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -DcomputeDevice=$DEVICE_OPT -Dtest=none -DforkMode=never -Dmaven.test.failure.ignore=true  -DfailIfNoTests=false -DwildcardSuites=$suite test
    elif [[ $suite == *"org.apache.spark.ml"* ]]; then
       echo
        echo Testing $suite ...
        echo
        mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -Dtest=none -DcomputeDevice=$DEVICE_OPT -DfailIfNoTests=false  -DwildcardSuites=$suite test
    else
      echo
      echo Testing java $suite ...
      echo
      mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -DcomputeDevice=$DEVICE_OPT -DwildcardSuites=none -Dtest=$suite test
    fi
  done
else
  SUBSUITE=$(echo $SUITE | tr "," "\n")
  for suite in ${SUBSUITE[*]}
  do
      if [[ $suite == *"com.intel.oap.mllib"* ]]; then
        echo
        echo Testing $suite ...
        echo
        mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -DcomputeDevice=$DEVICE_OPT -Dtest=none -DforkMode=never -Dmaven.test.failure.ignore=true  -DfailIfNoTests=false -DwildcardSuites=$suite test
      elif [[ $suite == *"org.apache.spark.ml"* ]]; then
         echo
          echo Testing $suite ...
          echo
          mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -DcomputeDevice=$DEVICE_OPT -Dtest=none -DwildcardSuites=$suite test
      else
        echo
        echo Testing java $suite ...
        echo
        mvn $MVN_NO_TRANSFER_PROGRESS -Dspark.version=$SPARK_VERSION -DcomputeDevice=$DEVICE_OPT -DwildcardSuites=none -Dtest=$suite test
      fi
  done
fi
