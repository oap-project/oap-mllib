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

if [[ -z $I_MPI_ROOT ]]; then
  echo I_MPI_ROOT not defined!
  exit 1
fi

if [[ -z $CCL_ROOT ]]; then
  echo CCL_ROOT not defined!
  exit 1
fi

# Use patchelf to change SONAME for libfabric
if [[ -z $(which patchelf) ]]; then
  echo Please install \"patchelf\"!
  exit 1
fi

if [[ $(basename $(pwd)) != "mllib-dal" ]]; then
  echo Please execute the script from \"mllib-dal\" directory!
  exit 1
fi

TARGET_DIR=./src/main/resources/lib

rm -f $TARGET_DIR/*.so*

cp $CCL_ROOT/lib/cpu_gpu_dpcpp/libccl.so.1.0 $TARGET_DIR/libccl.so.1


cp $I_MPI_ROOT/libfabric/lib/libfabric.so.1 $TARGET_DIR/libfabric.so.1
cp $I_MPI_ROOT/libfabric/lib/prov/libsockets-fi.so $TARGET_DIR

# Workaround dlopen (libfabric.so) in oneCCL
cp $I_MPI_ROOT/libfabric/lib/libfabric.so.1 $TARGET_DIR/libfabric.so
patchelf --set-soname libfabric.so $TARGET_DIR/libfabric.so

cp $I_MPI_ROOT/lib/release_mt/libmpi.so.12.0.0 $TARGET_DIR/libmpi.so.12

cp $DAALROOT/lib/intel64/libJavaAPI.so.1.1 $TARGET_DIR/libJavaAPI.so

cp $TBBROOT/lib/intel64/gcc4.8/libtbb.so.12.5 $TARGET_DIR/libtbb.so.12
cp $TBBROOT/lib/intel64/gcc4.8/libtbbmalloc.so.2.5 $TARGET_DIR/libtbbmalloc.so.2

echo oneAPI Toolkit version: $(basename $CCL_ROOT) > $TARGET_DIR/VERSION
