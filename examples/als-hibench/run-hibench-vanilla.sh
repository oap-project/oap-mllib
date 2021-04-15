#!/usr/bin/env bash

export HDFS_ROOT=hdfs://sr591:8020

SPARK_MASTER=yarn
SPARK_DRIVER_MEMORY=16G
SPARK_NUM_EXECUTORS=6
SPARK_EXECUTOR_CORES=28
SPARK_EXECUTOR_MEMORY_OVERHEAD=25G
SPARK_EXECUTOR_MEMORY=100G

SPARK_DEFAULT_PARALLELISM=$(expr $SPARK_NUM_EXECUTORS '*' $SPARK_EXECUTOR_CORES '*' 2)

# ======================================================= #

# for log suffix
SUFFIX=$( basename -s .sh "${BASH_SOURCE[0]}" )

# Check envs
if [[ -z $SPARK_HOME ]]; then
    echo SPARK_HOME not defined!
    exit 1
fi

if [[ -z $HADOOP_HOME ]]; then
    echo HADOOP_HOME not defined!
    exit 1
fi

export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop

APP_JAR=target/oap-mllib-examples-0.9.0-with-spark-3.0.0.jar
APP_CLASS=com.intel.hibench.sparkbench.ml.ALSExample

HDFS_INPUT=hdfs://sr591:8020/HiBench/ALS/Input
RANK=10
NUM_ITERATIONS=1
LAMBDA=0.1
IMPLICIT=true

/usr/bin/time -p $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER -v \
    --num-executors $SPARK_NUM_EXECUTORS \
    --driver-memory $SPARK_DRIVER_MEMORY \
    --executor-cores $SPARK_EXECUTOR_CORES \
    --executor-memory $SPARK_EXECUTOR_MEMORY \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.default.parallelism=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.sql.shuffle.partitions=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.driver.extraClassPath=$SPARK_DRIVER_CLASSPATH" \
    --conf "spark.executor.extraClassPath=$SPARK_EXECUTOR_CLASSPATH" \
    --conf "spark.shuffle.reduceLocality.enabled=false" \
    --conf "spark.executor.memoryOverhead=$SPARK_EXECUTOR_MEMORY_OVERHEAD" \
    --conf "spark.network.timeout=1200s" \
    --conf "spark.task.maxFailures=1" \
    --class $APP_CLASS \
    $APP_JAR \
    --rank $RANK --numIterations $NUM_ITERATIONS --implicitPrefs $IMPLICIT --lambda $LAMBDA \
    --numProductBlocks $SPARK_DEFAULT_PARALLELISM --numUserBlocks $SPARK_DEFAULT_PARALLELISM \
    $HDFS_INPUT \
    2>&1 | tee ALS-$SUFFIX-$(date +%m%d_%H_%M_%S).log

