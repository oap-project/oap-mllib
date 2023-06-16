#!/usr/bin/env bash

exampleDirs=(kmeans)

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ==========================
  echo Running $dir ...
  echo ==========================
  echo
  ./run.sh
  cd ..
done
