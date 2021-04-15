#!/usr/bin/env bash

export HADOOP_HOME=~/opt/hadoop-2.7.7
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HADOOP_HOME/lib/native

export SPARK_HOME=~/opt/spark-3.0.0-bin-hadoop2.7
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_PYTHON=python3

export PATH=$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH