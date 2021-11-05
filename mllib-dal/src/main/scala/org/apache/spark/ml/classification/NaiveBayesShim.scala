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

package org.apache.spark.ml.classification

import org.apache.spark.SPARK_VERSION
import org.apache.spark.ml.classification.{NaiveBayes => SparkNaiveBayes}
import org.apache.spark.ml.classification.spark320.{NaiveBayes => NaiveBayesSpark320}
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging

object NaiveBayesShim extends Logging {

  def create(uid: String) : SparkNaiveBayes = {

    logInfo(s"Loading NaiveBayes for Spark $SPARK_VERSION")

    val naiveBayes = SPARK_VERSION match {
      case "3.1.1" | "3.1.2" | "3.2.0" => new NaiveBayesSpark320(uid)
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    naiveBayes
  }

}

