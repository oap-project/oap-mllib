#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# install level-zero
$GITHUB_WORKSPACE/dev/install-level-zero-ubuntu.sh

sudo apt install  level-zero

#$GITHUB_WORKSPACE/dev/install-build-level-zero-deps-ubuntun.sh

$GITHUB_WORKSPACE/dev/install-level-zero-ubuntu.sh

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup building envs
source /opt/intel/oneapi/setvars.sh

cd  $GITHUB_WORKSPACE/dev/tools/list-compute-devices/
./build.sh
./run.sh

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-build-deps-gpu.sh
./build.sh -p CPU_GPU_PROFILE -q

unset LD_LIBRARY_PATH
./test.sh -p CPU_GPU_PROFILE -q -d host
