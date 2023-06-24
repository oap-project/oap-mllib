#!/usr/bin/env bash

exampleDirs=(kmeans pca linear-regression correlation summarizer)

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
