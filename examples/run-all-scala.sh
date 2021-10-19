#!/usr/bin/env bash

exampleDirs=(kmeans pca als naive-bayes linear-regression correlation)

for dir in ${exampleDirs[*]}
do
  cd $dir
  ./run.sh
  cd ..
done
