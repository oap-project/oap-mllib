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

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL}
import org.apache.spark.BarrierTaskContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector}
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.sql.Dataset

class NaiveBayesDALModel private[mllib] (
    val uid: String,
    val pi: Vector,
    val theta: Matrix,
    val sigma: Matrix
)

class NaiveBayesDALImpl(val uid: String,
                        val classNum: Int,
                        val executorNum: Int,
                        val executorCores: Int
                       ) extends Serializable with Logging {
  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): NaiveBayesDALModel = {

    val kvsIPPort = getOneCCLIPPort(labeledPoints.rdd)

    val labeledPointsTables = if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
      OneDAL.coalesceLabelPointsToNumericTables(labeledPoints, labelCol, featuresCol, executorNum)
    } else {
      OneDAL.coalesceSparseLabelPointsToSparseNumericTables(labeledPoints,
        labelCol, featuresCol, executorNum)
    }

    val results = labeledPointsTables.barrier().mapPartitionsWithIndex {
      case (rank: Int, tables: Iterator[(Long, Long)]) =>
        val context = BarrierTaskContext.get()
        val (featureTabAddr, lableTabAddr) = tables.next()

        OneCCL.init(executorNum, context.partitionId(), kvsIPPort)

        val computeStartTime = System.nanoTime()

        val result = new NaiveBayesResult
        cNaiveBayesDALCompute(featureTabAddr, lableTabAddr,
          classNum, executorNum, executorCores, result)

        val computeEndTime = System.nanoTime()

        val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

        println(s"NaiveBayesDAL compute took ${durationCompute} secs")

        val ret = if (context.partitionId() == 0) {
          val convResultStartTime = System.nanoTime()

          val pi = OneDAL.numericTableNx1ToVector(OneDAL.makeNumericTable(result.getPiNumericTable))
          val theta = OneDAL.numericTableToMatrix(
            OneDAL.makeNumericTable(result.getThetaNumericTable))

          val convResultEndTime = System.nanoTime()

          val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

          println(s"NaiveBayesDAL result conversion took ${durationCovResult} secs")

          Iterator((pi, theta))

        } else {
          Iterator.empty
        }
        context.barrier()
        OneCCL.cleanup()
        ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    val result = results(0)

    val model = new NaiveBayesDALModel(uid,
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
