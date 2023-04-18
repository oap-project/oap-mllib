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

package org.apache.spark.mllib.stat.spark321

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.stat.{SummarizerDALImpl, SummarizerShim}
import scala.annotation.varargs

import org.apache.spark.annotation.Since
import org.apache.spark.api.java.{JavaDoubleRDD, JavaRDD}
import org.apache.spark.ml.stat._
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary
import org.apache.spark.mllib.stat.correlation.Correlations
import org.apache.spark.mllib.stat.test.{
  ChiSqTest,
  ChiSqTestResult,
  KolmogorovSmirnovTest,
  KolmogorovSmirnovTestResult
}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * API for statistical functions in MLlib.
 */
@Since("1.1.0")
class Statistics extends SummarizerShim {

   /**
    * Computes column-wise summary statistics for the input RDD[Vector].
    *
    * @param X an RDD[Vector] for which column-wise summary statistics are to be computed.
    * @return [[org.apache.spark.mllib.stat.MultivariateStatisticalSummary]]
    *        object containing column-wise summary statistics.
    */
  @Since("1.1.0")
  def colStats(X: RDD[Vector]): MultivariateStatisticalSummary = {
    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      X.sparkContext)
    if (Utils.isOAPEnabled() && isPlatformSupported) {
      val handlePersistence = (X.getStorageLevel == StorageLevel.NONE)
      if (handlePersistence) {
        X.persist(StorageLevel.MEMORY_AND_DISK)
      }
      val rdd = X.map {
        v => v.asML
      }
      val executor_num = Utils.sparkExecutorNum(X.sparkContext)
      val executor_cores = Utils.sparkExecutorCores()
      val summary = new SummarizerDALImpl(executor_num, executor_cores)
        .computeSummarizerMatrix(rdd)
      if (handlePersistence) {
        X.unpersist()
      }
      summary
    } else {
      new RowMatrix(X).computeColumnSummaryStatistics()
    }
  }
}
