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
import org.apache.spark.ml.linalg.{Matrices, Vector}
import org.apache.spark.ml.util.{Instrumentation, OneCCL, OneDAL}
import org.apache.spark.ml.util.Utils.getOneCCLIPPort
import org.apache.spark.rdd.RDD

class NaiveBayesDALImpl(val uid: String,
                        val classNum: Int,
                        val executorNum: Int,
                        val executorCores: Int
                    ) extends Serializable with Logging {
  def train(labeledPoints: RDD[(Vector, Double)],
            instr: Option[Instrumentation]): NaiveBayesModel = {

    val kvsIPPort = getOneCCLIPPort(labeledPoints)

//    val labeledPointsTables = OneDAL.rddLabeledPointToMergedTables_repartition(labeledPoints, executorNum)
    val labeledPointsTables = OneDAL.rddLabeledPointToMergedTables(labeledPoints, executorNum)

    val results = labeledPointsTables.mapPartitionsWithIndex {
      case (rank: Int, tables: Iterator[(Long, Long)]) =>
        val (featureTabAddr, lableTabAddr) = tables.next()

        OneCCL.init(executorNum, rank, kvsIPPort)

        val result = new NaiveBayesResult
        cNaiveBayesDALCompute(featureTabAddr, lableTabAddr,
          classNum, executorNum, executorCores, result)

        val ret = if (OneCCL.isRoot()) {
          val pi = OneDAL.numericTableNx1ToVector(OneDAL.makeNumericTable(result.piNumericTable))
          val theta = OneDAL.numericTableToMatrix(OneDAL.makeNumericTable(result.thetaNumericTable))
          Iterator((pi, theta))
        } else {
          Iterator.empty
        }

        OneCCL.cleanup()
        ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    val result = results(0)

    val model = new NaiveBayesModel(uid,
      result._1,
      result._2,
      Matrices.zeros(0, 0))
    model
  }

  @native private def cNaiveBayesDALCompute(features: Long, labels: Long,
                                            class_num: Int,
                                            executor_num: Int,
                                            executor_cores: Int,
                                            result: NaiveBayesResult): Unit
}
