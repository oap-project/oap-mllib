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

if [[ -z $SPARK_VER ]]; then
 echo SPARK_VER not defined, using default.
else
 SPARK_VER=$SPARK_VER
fi

echo === Building Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo SPARK_VER=$SPARK_VER
echo =============================

cd $GITHUB_WORKSPACE/mllib-dal
if [[ -z $SPARK_VER ]]; then
    mvn --no-transfer-progress -DskipTests clean package
else
    mvn --no-transfer-progress -DskipTests clean package -P$SPARK_VER
fi
