#!/usr/bin/env bash

if [ ! -f /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  cd /tmp
  tee > /tmp/oneAPI.repo << EOF
[oneAPI]
name=Intel(R) oneAPI repository
baseurl=https://yum.repos.intel.com/oneapi
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
EOF
  sudo mv /tmp/oneAPI.repo /etc/yum.repos.d  
  # sudo yum groupinstall -y "Development Tools"
  # sudo yum install -y cmake
  sudo yum install -y intel-oneapi-dpcpp-cpp-2021.2.0 intel-oneapi-dal-devel-2021.2.0 intel-oneapi-tbb-devel-2021.2.0
else
  echo "oneAPI components already installed!"
fi  

echo "Building oneCCL ..."
cd /tmp
rm -rf oneCCL
git clone https://github.com/oneapi-src/oneCCL
cd oneCCL
git checkout 2021.2
mkdir build && cd build
cmake ..
make -j 2 install