#!/usr/bin/env bash

if [ ! -d /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  sudo apt clean
  cd /tmp
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt update
  sudo apt-get install -y intel-oneapi-ccl-devel-2021.16 \
                          intel-oneapi-tbb-common-devel-2021.13 intel-oneapi-tbb-devel-2022.2 \
                          intel-oneapi-mpi-devel-2021.16 \
                          intel-oneapi-dal-common-devel-2025.6 intel-oneapi-dal-devel-2025.6 \
                          intel-oneapi-compiler-dpcpp-cpp-2025.3 intel-oneapi-compiler-dpcpp-cpp-common-2025.3 intel-oneapi-compiler-dpcpp-cpp-runtime-2025.3 intel-oneapi-dpcpp-cpp-2025.3
else
  echo "oneAPI components already installed!"
fi
