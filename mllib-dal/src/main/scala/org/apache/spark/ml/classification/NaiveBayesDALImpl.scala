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

import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.{Instrumentation, OneDAL}
import org.apache.spark.rdd.RDD

class NaiveBayesDALImpl(val classNum: Int,
                        val executorNum: Int,
                        val executorCores: Int
                    ) extends Serializable with Logging {
  def train(features: RDD[Vector], labels: RDD[Double],
            instr: Option[Instrumentation]): NaiveBayesModel = {

    val featureTables = OneDAL.vectorsToMergedNumericTables(features, executorNum)
    val labelTables = OneDAL.doublesToNumericTables(labels, executorNum)

    featureTables.zip(labelTables).mapPartitions {
      case (tables: Iterator[(Long, Long)]) =>
      val (featureTabAddr, lableTabAddr) = tables.next()

      val result = new NaiveBayesResult
      cNaiveBayesDALCompute(featureTabAddr, lableTabAddr,
        classNum, executorNum, executorCores, result)

      Iterator()
    }

    null
  }

  @native private def cNaiveBayesDALCompute(features: Long, labels: Long,
                                            class_num: Int,
                                            executor_num: Int,
                                            executor_cores: Int,
                                            result: NaiveBayesResult): Unit
}
