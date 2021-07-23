#!/usr/bin/env bash

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup password-less & python3
$GITHUB_WORKSPACE/dev/test-cluster/config-ssh.sh
$GITHUB_WORKSPACE/dev/test-cluster/setup-python3.sh

# Setup cluster and envs
source $GITHUB_WORKSPACE/dev/test-cluster/setup-cluster.sh
