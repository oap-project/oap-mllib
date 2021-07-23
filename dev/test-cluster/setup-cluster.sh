#!/usr/bin/env bash

WORK_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $WORK_DIR

echo JAVA_HOME is $JAVA_HOME

[ -d ~/opt ] || mkdir ~/opt
cd ~/opt
[ -f spark-3.0.0-bin-hadoop2.7.tgz ] || wget --no-verbose https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz
[ -d spark-3.0.0-bin-hadoop2.7 ] || tar -xzf spark-3.0.0-bin-hadoop2.7.tgz
[ -f hadoop-2.7.7.tar.gz ] || wget --no-verbose https://archive.apache.org/dist/hadoop/core/hadoop-2.7.7/hadoop-2.7.7.tar.gz
[ -d hadoop-2.7.7 ] || tar -xzf hadoop-2.7.7.tar.gz

cd $WORK_DIR

HOST_IP=$(hostname -f)

sed -i "s/localhost/$HOST_IP/g" core-site.xml
sed -i "s/localhost/$HOST_IP/g" yarn-site.xml

cp ./core-site.xml ~/opt/hadoop-2.7.7/etc/hadoop/
cp ./hdfs-site.xml ~/opt/hadoop-2.7.7/etc/hadoop/
cp ./yarn-site.xml ~/opt/hadoop-2.7.7/etc/hadoop/
cp ./hadoop-env.sh ~/opt/hadoop-2.7.7/etc/hadoop/
cp ./spark-defaults.conf ~/opt/spark-3.0.0-bin-hadoop2.7/conf

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
