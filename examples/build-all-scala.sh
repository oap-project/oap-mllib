#!/usr/bin/env bash

exampleDirs=(kmeans)

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ===================
  echo Building $dir ...
  echo ===================
  echo
  ./build.sh
  cd ..
done
