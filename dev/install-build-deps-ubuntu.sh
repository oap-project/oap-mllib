#!/usr/bin/env bash

if [ ! -d /opt/intel/oneapi ]; then
  echo "Installing oneAPI components ..."
  cd /tmp
  # download the key to system keyring
  wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

  # add signed entry to apt sources and configure the APT client to use Intel repository:
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
  sudo apt-get update
  # sudo apt-get install -y build-essential cmake
  sudo apt-get install -y intel-basekit
else
  echo "oneAPI components already installed!"
fi
