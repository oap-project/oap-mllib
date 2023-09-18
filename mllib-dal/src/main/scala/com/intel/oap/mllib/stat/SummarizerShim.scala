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

import com.intel.oap.mllib.Utils

import org.apache.spark.{SPARK_VERSION, SparkException}
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.mllib.stat.spark333.{Statistics => SummarizerSpark333}

trait SummarizerShim extends Serializable with Logging {
  def colStats(X: RDD[Vector]): MultivariateStatisticalSummary

  }

object SummarizerShim extends Logging {
  def create(): SummarizerShim = {
    logInfo(s"Loading Summarizer for Spark $SPARK_VERSION")
    val summarizer = Utils.getSparkVersion() match {
      case "3.1.1" | "3.1.2" | "3.1.3" | "3.2.0" | "3.2.1" | "3.2.2" | "3.3.3" =>
        new SummarizerSpark333()
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    summarizer
  }
}
