#!/usr/bin/env bash

exampleDirs=(kmeans-pyspark pca-pyspark als-pyspark random-forest-pyspark)

cd python

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ==========================
  echo Running $dir ...
  echo ==========================
  echo
  ./run-gpu.sh
  cd ..
done

cd ..
