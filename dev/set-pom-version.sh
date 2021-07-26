#!/usr/bin/env bash

SPARK_VERSION=3.1.1
OAP_VERSION=1.2.0

mvn versions:set -DnewVersion=$OAP_VERSION
mvn versions:set-property -Dproperty=spark.version -DnewVersion=$SPARK_VERSION
