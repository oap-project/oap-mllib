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
package com.intel.oap.mllib.classification

import com.intel.oap.mllib.Utils
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.{SPARK_VERSION, SparkException}
import org.apache.spark.ml.classification.spark321.{RandomForestClassifier => RandomForestClassifier321}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset

trait RandomForestClassifierShim extends Logging {
  def initShim(params: ParamMap): Unit
  def train(dataset: Dataset[_], rfcTimer: RandomForestClassifierTimerClass): RandomForestClassificationModel
}

object RandomForestClassifierShim extends Logging {
  def create(uid: String): RandomForestClassifierShim = {
    logInfo(s"Loading RandomForestClassifier for Spark $SPARK_VERSION")

    val shim = Utils.getSparkVersion() match {
      case "3.1.1" | "3.1.2" | "3.1.3" | "3.2.0" | "3.2.1" => new RandomForestClassifier321(uid)
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    shim
  }
}
