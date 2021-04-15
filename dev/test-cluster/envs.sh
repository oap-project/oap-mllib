# Set user Spark and Hadoop home directory
export HADOOP_HOME=~/opt/hadoop-2.7.7
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_HOME=~/opt/spark-3.0.0-bin-hadoop2.7

export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PYSPARK_PYTHON=python3

# Set user HDFS Root
export HDFS_ROOT=hdfs://localhost:8020
export OAP_MLLIB_DATA_ROOT=OAPMLlib/Data
# Set user Intel MLlib Root directory
export OAP_MLLIB_ROOT=${GITHUB_WORKSPACE}

# Target jar built
OAP_MLLIB_JAR_NAME=oap-mllib-1.1.0.jar
OAP_MLLIB_JAR=$OAP_MLLIB_ROOT/mllib-dal/target/$OAP_MLLIB_JAR_NAME

# Use absolute path
SPARK_DRIVER_CLASSPATH=$OAP_MLLIB_JAR
# Use relative path
SPARK_EXECUTOR_CLASSPATH=./$OAP_MLLIB_JAR_NAME
