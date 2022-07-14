#!/usr/bin/env bash
echo "Installing level-zero components ..."
sudo apt-get install -y gpg-agent wget
wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
sudo apt-key add -
 sudo apt-add-repository \
'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
sudo apt-get install -y \
intel-opencl-icd \
intel-level-zero-gpu level-zero 
