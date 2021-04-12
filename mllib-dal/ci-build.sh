#!/usr/bin/env bash

# Check envs for building
if [[ -z $JAVA_HOME ]]; then
 echo $JAVA_HOME not defined!
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

if [[ -z $I_MPI_ROOT ]]; then
 echo I_MPI_ROOT not defined!
 exit 1
fi

echo === Building Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo MPI_ROOT=$I_MPI_ROOT
echo Clang Version: $(clang -dumpversion)
echo =============================

mvn --no-transfer-progress -DskipTests clean package
