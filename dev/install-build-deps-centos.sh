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
  sudo yum install -y intel-basekit-getting-started intel-oneapi-advisor intel-oneapi-ccl-2021.8.0 intel-oneapi-ccl-devel-2021.8.0 intel-oneapi-common-licensing-2023.0.0  \
                      intel-oneapi-tbb-2021.8.0 intel-oneapi-tbb-common-2021.8.0 intel-oneapi-tbb-common-devel-2021.8.0 intel-oneapi-tbb-devel-2021.8.0 \
                      intel-oneapi-mpi-2021.8.0 intel-oneapi-mpi-devel-2021.8.0 \
                      intel-oneapi-dal-2023.0.0 intel-oneapi-dal-common-2023.0.0 intel-oneapi-dal-common-devel-2023.0.0 intel-oneapi-dal-devel-2023.0.0 \
                      intel-oneapi-compiler-dpcpp-cpp-2023.0.0 intel-oneapi-compiler-dpcpp-cpp-common-2023.0.0 intel-oneapi-compiler-dpcpp-cpp-runtime-2023.0.0 intel-oneapi-dpcpp-cpp-2023.0.0
else
  echo "oneAPI components already installed!"
fi  
