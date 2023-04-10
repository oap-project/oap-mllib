#!/usr/bin/env bash

# exit when any command fails
set -e

# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command filed with exit code $?."' EXIT

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo JAVA_HOME is $JAVA_HOME

# Setup password-less & python3
$GITHUB_WORKSPACE/dev/test-cluster/config-ssh.sh
$GITHUB_WORKSPACE/dev/test-cluster/setup-python3.sh

# setup envs
source $SCRIPT_DIR/load-spark-envs.sh

# download spark & hadoop bins
[ -d ~/opt ] || mkdir ~/opt
cd ~/opt
[ -f spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz ] || wget --no-verbose https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz
[ -d spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION ] || tar -xzf spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION.tgz

cd $SCRIPT_DIR

HOST_IP=$(hostname -f)

sed -i "s/localhost/$HOST_IP/g" core-site.xml

cp ../log4j.properties ~/opt/spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION/conf
cp ./spark-defaults.conf ~/opt/spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION/conf
cp ./spark-env.sh ~/opt/spark-$SPARK_VERSION-bin-$SPARK_HADOOP_VERSION/conf

echo $HOST_IP > $SPARK_HOME/conf/slaves

ls -l $SPARK_HOME/conf

#start spark standalone
$SPARK_HOME/sbin/start-all.sh
