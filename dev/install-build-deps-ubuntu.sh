#!/usr/bin/env bash

if [ ! -d /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  cd /tmp
  wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
  sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
  rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
  echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt-get update
  sudo apt-get install -y intel-oneapi-dpcpp-cpp-2021.2.0 intel-oneapi-dal-devel-2021.2.0 intel-oneapi-tbb-devel-2021.2.0 intel-oneapi-ccl-devel-2021.2.0
else
  echo "oneAPI components already installed!"
fi  
