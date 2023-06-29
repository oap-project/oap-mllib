#!/usr/bin/env bash

exampleDirs=(kmeans-scala pca-scala linear-regression-scala correlation-scala summarizer-scala)

cd scala

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
