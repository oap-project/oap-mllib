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

# Check lib dependencies for building
RESOURCE_PATH=src/main/resources/lib
LIBS=(libccl.so.1 libfabric.so libfabric.so.1 libJavaAPI.so libmpi.so.12 \
  libsockets-fi.so libtbbmalloc.so.2 libtbb.so.12)
for lib in ${LIBS[@]}
do
  if [[ ! -f ./$RESOURCE_PATH/$lib ]]; then
    echo $RESOURCE_PATH/$lib does not exsit, please run ../dev/prepare-build-deps.sh!
    exit 1
fi
done

if [[ -f ./$RESOURCE_PATH/libsycl.so.5 ]]; then
 echo GPU libs found! Please re-run ../dev/prepare-build-deps.sh!
 exit 1
fi

versionArray=(
  spark-3.0.0 \
  spark-3.0.1 \
  spark-3.0.2 \
  spark-3.1.1
)

SPARK_VER=spark-3.1.1
MVN_NO_TRANSFER_PROGRESS=

print_usage() {
  echo
  echo Usage: ./build.sh [-p spark-x.x.x] [-q] [-h]
  echo
  echo Supported Spark versions:
  for version in ${versionArray[*]}
  do
    echo "    $version"
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

if [[ ! ${versionArray[*]} =~ $SPARK_VER ]]; then
  echo Error: $SPARK_VER version is not supported!
  exit 1
fi

print_usage

export PLATFORM_PROFILE=CPU_ONLY_PROFILE

echo === Building Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo Spark Version: $SPARK_VER
echo Platform Profile: $PLATFORM_PROFILE
echo =============================
echo
echo Building with $SPARK_VER ...
echo
mvn $MVN_NO_TRANSFER_PROGRESS -P$SPARK_VER -DskipTests clean package
