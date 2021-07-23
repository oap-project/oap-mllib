#!/usr/bin/env bash

source ../../conf/env.sh

# Data file is from Spark Examples (data/mllib/sample_kmeans_data.txt) and put in examples/data
# The data file should be copied to $HDFS_ROOT before running examples
DATA_FILE=data/sample_kmeans_data.txt

APP_JAR=target/oap-mllib-examples-$OAP_MLLIB_VERSION.jar
APP_CLASS=org.apache.spark.examples.ml.KMeansExample

time $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER -v \
    --num-executors $SPARK_NUM_EXECUTORS \
    --driver-memory $SPARK_DRIVER_MEMORY \
    --total-executor-cores $SPARK_TOTAL_CORES \
    --executor-cores $SPARK_EXECUTOR_CORES \
    --executor-memory $SPARK_EXECUTOR_MEMORY \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.default.parallelism=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.sql.shuffle.partitions=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.driver.extraClassPath=$SPARK_DRIVER_CLASSPATH" \
    --conf "spark.executor.extraClassPath=$SPARK_EXECUTOR_CLASSPATH" \
    --conf "spark.oap.mllib.useGPU=true" \
    --conf "spark.executor.resource.gpu.amount=1" \
    --conf "spark.task.resource.gpu.amount=1" \
    --conf "spark.shuffle.reduceLocality.enabled=false" \
    --conf "spark.network.timeout=1200s" \
    --conf "spark.task.maxFailures=1" \
    --jars $OAP_MLLIB_JAR \
    --class $APP_CLASS \
    $APP_JAR $DATA_FILE \
    2>&1 | tee KMeans-$(date +%m%d_%H_%M_%S).log

