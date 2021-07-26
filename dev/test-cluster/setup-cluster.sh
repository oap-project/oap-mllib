#!/usr/bin/env bash

WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $WORK_DIR

echo JAVA_HOME is $JAVA_HOME

HADOOP_VERSION=3.2.0
SPARK_VERSION=3.1.1
SPARK_HADOOP_VERSION=hadoop3.2

[ -d ~/opt ] || mkdir ~/opt
cd ~/opt
[ -f $SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz ] || wget --no-verbose https://archive.apache.org/dist/spark/$SPARK_VERSION/$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz
[ -d $SPARK_VERSION-bin-$SPARK_HADOOP_VERSION ] || tar -xzf $SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz
[ -f hadoop-$HADOOP_VERSION.tar.gz ] || wget --no-verbose https://archive.apache.org/dist/hadoop/core/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz
[ -d hadoop-$HADOOP_VERSION ] || tar -xzf hadoop-$HADOOP_VERSION.tar.gz

cd $WORK_DIR

HOST_IP=$(hostname -f)

sed -i "s/localhost/$HOST_IP/g" core-site.xml
sed -i "s/localhost/$HOST_IP/g" yarn-site.xml

cp ./core-site.xml ~/opt/hadoop-$HADOOP_VERSION/etc/hadoop/
cp ./hdfs-site.xml ~/opt/hadoop-$HADOOP_VERSION/etc/hadoop/
cp ./yarn-site.xml ~/opt/hadoop-$HADOOP_VERSION/etc/hadoop/
cp ./hadoop-env.sh ~/opt/hadoop-$HADOOP_VERSION/etc/hadoop/
cp ./spark-defaults.conf ~/opt/$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION/conf

source ./setup-spark-envs.sh

echo $HOST_IP > $HADOOP_HOME/etc/hadoop/slaves
echo $HOST_IP > $SPARK_HOME/conf/slaves

# create directories
mkdir -p /tmp/run/hdfs/namenode
mkdir -p /tmp/run/hdfs/datanode

# hdfs format
$HADOOP_HOME/bin/hdfs namenode -format

# start hdfs and yarn
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh

hadoop fs -ls /
yarn node -list
