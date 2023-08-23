#!/usr/bin/env bash

exampleDirs=(kmeans-pyspark pca-pyspark als-pyspark \
     correlation-pyspark summarizer-pyspark)

cd python

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ==========================
  echo Running $dir ...
  echo ==========================
  echo
  ./run-cpu.sh
  cd ..
done

cd ..
