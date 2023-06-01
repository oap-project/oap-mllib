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

check_build_deps() {
  # Check lib dependencies for building
  RESOURCE_PATH=src/main/resources/lib
  LIBS=(libJavaAPI.so libtbbmalloc.so.2 libtbb.so.12)
  for lib in ${LIBS[@]}
  do
    if [[ ! -f ./$RESOURCE_PATH/$lib ]]; then
      echo \"$RESOURCE_PATH/$lib\" does not exsit, please run \"../dev/prepare-build-deps.sh\"!
      exit 1
  fi
  done
}

MVN_NO_TRANSFER_PROGRESS=

print_usage() {
  echo
  echo "Usage: ./build.sh [-p <CPU_GPU_PROFILE | CPU_ONLY_PROFILE>] [-q] [-h]"
  echo
  echo "-p  Supported Platform Profiles:"
    echo "    CPU_GPU_PROFILE (Default)"
    echo "    CPU_ONLY_PROFILE"
  echo
}

while getopts "p:qh" opt
do
case $opt in
  p) PLATFORM_OPT=$OPTARG ;;
  q) MVN_NO_TRANSFER_PROGRESS=--no-transfer-progress ;;
  h | *)
     print_usage
     exit 1
     ;;
esac
done

shift "$((OPTIND-1))"

SUITE=$1

print_usage

if [[ -n $PLATFORM_OPT && ! ($PLATFORM_OPT == CPU_GPU_PROFILE || $PLATFORM_OPT == CPU_ONLY_PROFILE) ]]; then
  echo
  echo Platform Profile should be CPU_GPU_PROFILE or CPU_ONLY_PROFILE, but \"$PLATFORM_OPT\" found!
  echo
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
# Set default PLATFORM_PROFILE from RELEASE envs
export PLATFORM_PROFILE=${PLATFORM_OPT:-$PLATFORM_PROFILE}

# Check Build deps
check_build_deps

echo
echo === Building Environments ===
echo Platform Profile: $PLATFORM_PROFILE
echo Spark Version: $SPARK_VERSION
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
if [[ $PLATFORM_PROFILE == CPU_GPU_PROFILE ]]
then
  echo ICPX Version: $(icpx -dumpversion)
elif [[ $PLATFORM_PROFILE == CPU_ONLY_PROFILE ]]
then
  echo GCC Version: $(gcc -dumpversion)
fi
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo =============================
echo

mvn $MVN_NO_TRANSFER_PROGRESS -DskipTests clean package
