#!/usr/bin/env bash

# Copyright 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WORK_DIR="$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )"

DAAL_JAR=${ONEAPI_ROOT}/dal/latest/lib/onedal.jar

if [ ! -f "$DAAL_JAR" ]; then
    echo \$DAAL_JAR does not exist!
    exit 1
fi

if [[ ! -e "$SPARK_HOME" ]]; then
	echo \$SPARK_HOME does not exist!
    exit 1	
fi

javah -d $WORK_DIR/javah -classpath "$WORK_DIR/../../../target/classes:$DAAL_JAR:$SPARK_HOME/jars/*" -force \
    com.intel.oap.mllib.OneCCL$ \
    com.intel.oap.mllib.OneDAL$ \
    com.intel.oap.mllib.clustering.KMeansDALImpl \
    com.intel.oap.mllib.feature.PCADALImpl \
    com.intel.oap.mllib.recommendation.ALSDALImpl \
    com.intel.oap.mllib.classification.NaiveBayesDALImpl \
    com.intel.oap.mllib.regression.LinearRegressionDALImpl \
    com.intel.oap.mllib.stat.CorrelationDALImpl \
    com.intel.oap.mllib.classification.RandomForestClassifierDALImpl \
    com.intel.oap.mllib.regression.RandomForestRegressorDALImpl \
    com.intel.oap.mllib.stat.SummarizerDALImpl \
    com.intel.oneapi.dal.table.HomogenTableImpl \
    com.intel.oneapi.dal.table.SimpleMetadataImpl \
    com.intel.oneapi.dal.table.ColumnAccessor \
    com.intel.oneapi.dal.table.RowAccessor
