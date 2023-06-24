#!/usr/bin/env bash

exampleDirs=(kmeans-scala pca-scala als-scala naive-bayes-scala \
       	linear-regression-scala correlation-scala summarizer-scala)

cd scala

for dir in ${exampleDirs[*]}
do
  cd $dir
  echo
  echo ===================
  echo Building $dir ...
  echo ===================
  echo
  ./build.sh
  cd ..
done

cd ..
