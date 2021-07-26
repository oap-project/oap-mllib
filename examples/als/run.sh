#!/usr/bin/env bash

source ../../conf/env.sh

# Data file is converted from oneDAL examples ($DAALROOT/examples/daal/data/batch/implicit_als_csr.csv)
# The data file should be copied to $HDFS_ROOT before running examples
DATA_FILE=data/onedal_als_csr_ratings.txt

APP_JAR=target/oap-mllib-examples-$OAP_MLLIB_VERSION.jar
APP_CLASS=org.apache.spark.examples.ml.ALSExample

time $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER -v \
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
    --conf "spark.network.timeout=1200s" \
    --conf "spark.task.maxFailures=1" \
    --jars $OAP_MLLIB_JAR \
    --class $APP_CLASS \
    $APP_JAR $DATA_FILE \
    2>&1 | tee ALS-$(date +%m%d_%H_%M_%S).log
