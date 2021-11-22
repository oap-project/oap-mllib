#!/usr/bin/env bash

exampleDirs=(kmeans-pyspark pca-pyspark als-pyspark)

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
