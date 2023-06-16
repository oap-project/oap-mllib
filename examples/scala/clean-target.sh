#!/usr/bin/env bash

exampleDirs=(kmeans pca als naive-bayes linear-regression correlation summarizer)

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ==========================
  echo Cleaning $dir ...
  echo ==========================
  echo
  rm -rf ./target/
  cd ..
done
