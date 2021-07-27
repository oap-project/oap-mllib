#!/usr/bin/env bash

exampleDirs=(kmeans-pyspark pca-pyspark als-pyspark)

for dir in ${exampleDirs[*]}
do
  cd $dir
  ./run.sh
  cd ..
done
