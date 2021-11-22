#!/usr/bin/env bash

exampleDirs=(kmeans pca als naive-bayes linear-regression correlation)

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
