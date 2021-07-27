#!/usr/bin/env bash

exampleDirs=(kmeans pca als naive-bayes linear-regression)

for dir in ${exampleDirs[*]}
do
  cd $dir
  ./build.sh
  cd ..
done
