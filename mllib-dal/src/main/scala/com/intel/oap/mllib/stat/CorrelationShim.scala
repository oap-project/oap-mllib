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

package com.intel.oap.mllib.stat

import org.apache.spark.{SPARK_VERSION, SparkException}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.storage.StorageLevel

import scala.reflect.ClassTag

import org.apache.spark.ml.stat.spark321.{Correlation => CorrelationSpark321 }

trait CorrelationShim extends Serializable with Logging {
  def corr(dataset: Dataset[_], column: String, method: String): DataFrame
}

object CorrelationShim extends Logging {
  def create(): CorrelationShim = {
    logInfo(s"Loading Correlation for Spark $SPARK_VERSION")
    val als = SPARK_VERSION match {
      case "3.1.1" | "3.1.2" | "3.1.3" | "3.2.0" | "3.2.1" => new CorrelationSpark321()
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    als
  }
}
