#!/usr/bin/env bash

if [ ! -d /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  sudo apt clean
  cd /tmp
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt update
  sudo apt-get install -y intel-oneapi-ccl-2021.8.0 intel-oneapi-dpcpp-cpp-2023.0.0 intel-oneapi-tbb-2021.8.0 intel-oneapi-dal-2023.0.0 intel-oneapi-mpi-2021.8.0
else
  echo "oneAPI components already installed!"
fi
