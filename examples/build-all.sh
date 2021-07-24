#!/usr/bin/env bash

for dir in kmeans pca als naive-bayes linear-regression
do
  cd $dir
  ./build.sh
  cd ..
done
