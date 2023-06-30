/*
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.oap.mllib.clustering

import com.intel.oap.mllib.Utils

import org.apache.spark.internal.Logging
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.clustering.spark322.{KMeans => KMeansSpark322}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.{SPARK_VERSION, SparkException}

trait KMeansShim extends Logging {
  def initShim(params: ParamMap): Unit
  def fit(dataset: Dataset[_]): KMeansModel
}

object KMeansShim extends Logging {
  def create(uid: String): KMeansShim = {
    logInfo(s"Loading KMeans for Spark $SPARK_VERSION")
    val kmeans = Utils.getSparkVersion() match {
      case "3.1.1" | "3.1.2" | "3.1.3" | "3.2.0" | "3.2.1" | "3.2.2" => new KMeansSpark322(uid)
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    kmeans
  }
}
