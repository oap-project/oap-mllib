#!/usr/bin/env bash

for dir in kmeans-pyspark pca-pyspark als-pyspark
do
  cd $dir
  ./run.sh
  cd ..
done
