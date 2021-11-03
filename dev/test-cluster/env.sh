# == OAP MLlib users to customize the following environments for running examples ======= #

# ============== Minimum Settings ============= #

# Import RELEASE envs
source $GITHUB_WORKSPACE/RELEASE
# Set OAP MLlib version (e.g. 1.1.0)
OAP_MLLIB_VERSION=${OAP_MLLIB_VERSION}
# Set Spark master
SPARK_MASTER=yarn
# Set Hadoop home path
export HADOOP_HOME=$HADOOP_HOME
# Set Spark home path
export SPARK_HOME=$SPARK_HOME
# Set HDFS Root, should be hdfs://xxx or file://xxx
export HDFS_ROOT=hdfs://localhost:8020
# Set OAP MLlib source code root directory
export OAP_MLLIB_ROOT=$GITHUB_WORKSPACE

# ============================================= #

# Set HADOOP_CONF_DIR for Spark
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop

# Set JAR name & path
OAP_MLLIB_JAR_NAME=oap-mllib-$OAP_MLLIB_VERSION.jar
OAP_MLLIB_JAR=$OAP_MLLIB_ROOT/mllib-dal/target/$OAP_MLLIB_JAR_NAME
# Set Spark driver & executor classpaths,
# absolute path for driver, relative path for executor
SPARK_DRIVER_CLASSPATH=$OAP_MLLIB_JAR
SPARK_EXECUTOR_CLASSPATH=./$OAP_MLLIB_JAR_NAME

# Set Spark resources, can be overwritten in example
SPARK_DRIVER_MEMORY=1G
SPARK_NUM_EXECUTORS=2
SPARK_EXECUTOR_CORES=1
SPARK_EXECUTOR_MEMORY=1G
SPARK_DEFAULT_PARALLELISM=$(expr $SPARK_NUM_EXECUTORS '*' $SPARK_EXECUTOR_CORES '*' 2)

# Checks

for dir in $SPARK_HOME $HADOOP_HOME $OAP_MLLIB_JAR
do
    if [[ ! -e $dir ]]; then
        echo $dir does not exist!
        exit 1
    fi
done
