#!/usr/bin/env bash

exampleDirs=(kmeans pca als naive-bayes linear-regression correlation summarizer)

for dir in ${exampleDirs[*]}
do
  mv $dir $dir-scala
  echo
  echo ===================
  echo Building $dir ...
  echo ===================
  echo
done
