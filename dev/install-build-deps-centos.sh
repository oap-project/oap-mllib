#!/usr/bin/env bash

if [ ! -f /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  cd /tmp
  tee > /tmp/oneAPI.repo << EOF
  [oneAPI]
  name=Intel® oneAPI repository
  baseurl=https://yum.repos.intel.com/oneapi
  enabled=1
  gpgcheck=1
  repo_gpgcheck=1
  gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  EOF
  sudo mv /tmp/oneAPI.repo /etc/yum.repos.d  
  # sudo yum groupinstall -y "Development Tools"
  # sudo yum install -y cmake
  sudo yum install intel-basekit
else
  echo "oneAPI components already installed!"
fi  
