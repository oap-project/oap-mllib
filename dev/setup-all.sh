#!/usr/bin/env bash

# Setup hosts
# Use second internal IP, use first IP will be SSH timeout
HOST_IP=$(hostname -I | cut -f2 -d" ")
echo $HOST_IP $(hostname) | sudo tee -a /etc/hosts

# Install dependencies for building
$GITHUB_WORKSPACE/dev/install-build-deps-ubuntu.sh

# Setup password-less & python3
$GITHUB_WORKSPACE/dev/test-cluster/config-ssh.sh
$GITHUB_WORKSPACE/dev/test-cluster/setup-python3.sh

# Setup cluster and envs
source $GITHUB_WORKSPACE/dev/test-cluster/setup-cluster.sh
