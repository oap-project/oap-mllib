#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup building envs
source /opt/intel/oneapi/setvars.sh --ccl-configuration=cpu

# Prepare lib resources
cd $GITHUB_WORKSPACE/mllib-dal
./build.sh -p CPU_ONLY_PROFILE -q

./test.sh -p CPU_ONLY_PROFILE -q -d CPU
