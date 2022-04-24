#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

lscpu

#$GITHUB_WORKSPACE/dev/install-build-level-zero-deps-ubuntun.sh
$GITHUB_WORKSPACE/dev/install-level-zero-ubuntu.sh

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup building envs
source /opt/intel/oneapi/setvars.sh
#source /home/runner/work/level-zero/setvars.sh

cd  $GITHUB_WORKSPACE/dev/tools/check-gpu-cpu/
./build.sh
./run.sh

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
../dev/prepare-build-deps-gpu.sh
./build.sh -p CPU_GPU_PROFILE

unset LD_LIBRARY_PATH
./test.sh -p CPU_GPU_PROFILE -t cpu -q
