#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup password-less & python3
$GITHUB_WORKSPACE/dev/test-cluster/config-ssh.sh
$GITHUB_WORKSPACE/dev/test-cluster/setup-python3.sh

# Setup cluster and envs
source $GITHUB_WORKSPACE/dev/test-cluster/setup-cluster.sh
