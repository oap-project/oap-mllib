#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

if [ -z ${ONEAPI_ROOT} ]; then
  echo Please source Intel oneAPI Toolkit environments!
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

if [[ $(basename $(pwd)) != "mllib-dal" ]]; then
  echo Please execute the script from \"mllib-dal\" directory!
  exit 1
fi

TARGET_DIR=./src/main/resources/lib

rm -f $TARGET_DIR/*.so*

# Copy oneDAL JavaAPI libs
cp $DAALROOT/lib/intel64/libJavaAPI.so $TARGET_DIR/libJavaAPI.so
cp $TBBROOT/lib/intel64/gcc4.8/libtbb.so.12 $TARGET_DIR/libtbb.so.12
cp $TBBROOT/lib/intel64/gcc4.8/libtbbmalloc.so.2 $TARGET_DIR/libtbbmalloc.so.2

echo oneAPI Toolkit version: $(basename $DAALROOT) > $TARGET_DIR/VERSION
exit 0
