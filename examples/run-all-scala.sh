#!/usr/bin/env bash

for dir in kmeans pca als
do
  cd $dir
  ./run.sh
  cd ..
done
