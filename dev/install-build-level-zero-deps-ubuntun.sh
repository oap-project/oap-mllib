#!/usr/bin/env bash
echo "Installing oneAPI components ..."
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero
git checkout v1.7.15
mkdir build
cd build
cmake ..
sudo cmake --build . --config Release
sudo cmake --build . --config Release --target package
sudo cmake --build . --config Release --target install
cd ../
echo -e "#!/bin/bash \n export L0T_ROOT=$(dirname $(realpath ${BASH_SOURCE})) \n export LD_LIBRARY_PATH=${L0T_ROOT}/build/lib:${LD_LIBRARY_PATH:-} " > setvars.sh
cat setvars.sh
