#!/usr/bin/env bash

# Setup hosts
echo "$(hostname -i) $(hostname)" | sudo tee -a /etc/hosts

# Install all building dependencies
${{github.workspace}}/dev/install-build-deps-ubuntu.sh

# Setup password-less & Python3
cd $GITHUB_WORKSPACE/dev/test-cluster
./config-ssh.sh
./setup-python3.sh

# Setup Hadoop cluster and envs
source ./setup-cluster.sh