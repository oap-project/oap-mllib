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

# Set default version
SPARK_VER=spark-3.0.0

while getopts "p:" opt
do
case $opt in
  p) SPARK_VER=$OPTARG ;;
  *) echo
     echo Usage: ./build.sh [-p spark-x.x.x]
     echo
     exit 1
     ;;
esac
done

echo
echo Usage: ./build.sh [-p spark-x.x.x]
echo
echo === Building Environments ===
echo JAVA_HOME=$JAVA_HOME
echo DAALROOT=$DAALROOT
echo TBBROOT=$TBBROOT
echo CCL_ROOT=$CCL_ROOT
echo Maven Version: $(mvn -v | head -n 1 | cut -f3 -d" ")
echo Clang Version: $(clang -dumpversion)
echo Spark Version: $SPARK_VER
echo =============================
echo Building with $SPARK_VER ...
mvn -P$SPARK_VER -DskipTests clean package
