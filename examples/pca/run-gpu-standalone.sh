#!/usr/bin/env bash

source ../../conf/env.sh

APP_JAR=target/oap-mllib-examples-$OAP_MLLIB_VERSION.jar
APP_CLASS=org.apache.spark.examples.ml.PCAExample

USE_GPU=true
RESOURCE_FILE=$PWD/IntelGpuResourceFile.json
WORKER_GPU_AMOUNT=4
EXECUTOR_GPU_AMOUNT=1
TASK_GPU_AMOUNT=1

# Should run in standalone mode
time $SPARK_HOME/bin/spark-submit --master $SPARK_MASTER \
    --num-executors $SPARK_NUM_EXECUTORS \
    --executor-cores $SPARK_EXECUTOR_CORES \
    --total-executor-cores $SPARK_TOTAL_CORES \
    --driver-memory $SPARK_DRIVER_MEMORY \
    --executor-memory $SPARK_EXECUTOR_MEMORY \
    --conf "spark.serializer=org.apache.spark.serializer.KryoSerializer" \
    --conf "spark.default.parallelism=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.sql.shuffle.partitions=$SPARK_DEFAULT_PARALLELISM" \
    --conf "spark.driver.extraClassPath=$SPARK_DRIVER_CLASSPATH" \
    --conf "spark.executor.extraClassPath=$SPARK_EXECUTOR_CLASSPATH" \
    --conf "spark.oap.mllib.useGPU=$USE_GPU" \
    --conf "spark.worker.resourcesFile=$RESOURCE_FILE" \
    --conf "spark.worker.resource.gpu.amount=$WORKER_GPU_AMOUNT" \
    --conf "spark.executor.resource.gpu.amount=$EXECUTOR_GPU_AMOUNT" \
    --conf "spark.task.resource.gpu.amount=$TASK_GPU_AMOUNT" \
    --conf "spark.shuffle.reduceLocality.enabled=false" \
    --conf "spark.network.timeout=1200s" \
    --conf "spark.task.maxFailures=1" \
    --jars $OAP_MLLIB_JAR \
    --class $APP_CLASS \
    $APP_JAR \
    2>&1 | tee PCA-$(date +%m%d_%H_%M_%S).log

