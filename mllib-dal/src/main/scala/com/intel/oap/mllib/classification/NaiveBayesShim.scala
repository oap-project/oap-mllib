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

import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.classification.spark320.{NaiveBayes => NaiveBayesSpark320}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.Dataset
import org.apache.spark.{SPARK_VERSION, SparkException}

trait NaiveBayesShim extends Logging {
  def initShim(params: ParamMap): Unit
  def train(dataset: Dataset[_]): NaiveBayesModel
}

object NaiveBayesShim extends Logging {
  def create(uid: String): NaiveBayesShim = {
    logInfo(s"Loading NaiveBayes for Spark $SPARK_VERSION")
    // For example: CHD spark version was 3.1.1.3.1.7290.5-2.
    // Intercepting the third decimal point is the spark version.
    val array = SPARK_VERSION.split("\\.")
    val sparkVersion = if (array.size > 3) {
      val version = array.take(3).mkString(".")
      version
    } else {
      SPARK_VERSION
    }
    val shim = sparkVersion match {
      case "3.1.1" | "3.1.2" | "3.2.0" => new NaiveBayesSpark320(uid)
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    shim
  }
}
