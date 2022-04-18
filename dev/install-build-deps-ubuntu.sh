#!/usr/bin/env bash

if [ ! -d /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  cd /tmp
  wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
  sudo apt-get update
  # sudo apt-get install -y build-essential cmake
  sudo apt-get install -y intel-basekit
else
  echo "oneAPI components already installed!"
fi
