#!/usr/bin/env bash
if [ ! -d /home/runner/work/level-zero ]; then
  echo "Installing level-zero components ..."
  cd /home/runner/work
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
  echo -e "#!/bin/bash \n export LD_LIBRARY_PATH=/usr/local/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH:-} " > setvars.sh
  cat setvars.sh
else
  echo "level-zero components already installed!"
fi
