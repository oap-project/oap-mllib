#!/usr/bin/env bash

source ../../conf/env.sh

APP_JAR=target/oap-mllib-examples-$OAP_MLLIB_VERSION-with-spark-3.0.0.jar
APP_CLASS=org.apache.spark.examples.ml.LinearRegressionExample
DATA_FILE=data/sample_linear_regression_data.txt

OAP_MLLIB_ENABLED=true

time $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER -v \
    --num-executors $SPARK_NUM_EXECUTORS \
    --driver-memory $SPARK_DRIVER_MEMORY \
    --executor-cores $SPARK_EXECUTOR_CORES \
    --executor-memory $SPARK_EXECUTOR_MEMORY \
    --conf "spark.oap.mllib.enabled=$OAP_MLLIB_ENABLED" \
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
    2>&1 | tee LinearRegression-$(date +%m%d_%H_%M_%S).log
