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

import com.intel.oap.mllib.{OneCCL, OneDAL}
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Vector}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalDALSummary, MultivariateStatisticalSummary => Summary}
import org.apache.spark.rdd.RDD
import com.intel.oap.mllib.Utils.getOneCCLIPPort


class SummarizerDALImpl(
                          val executorNum: Int,
                          val executorCores: Int)
  extends Serializable with Logging {

  def computeSummarizerMatrix(data: RDD[Vector]): Summary = {
    val kvsIPPort = getOneCCLIPPort(data)

    val sparkContext = data.sparkContext
    val useGPU = sparkContext.getConf.getBoolean("spark.oap.mllib.useGPU", false)

    val coalescedTables = OneDAL.rddVectorToMergedTables(data, executorNum)

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>

      val gpuIndices = if (useGPU) {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }

      val tableArr = table.next()
      OneCCL.init(executorNum, rank, kvsIPPort)

      val computeStartTime = System.nanoTime()

      val result = new SummarizerResult()
      cSummarizerTrainDAL(
        tableArr,
        executorNum,
        executorCores,
        useGPU,
        gpuIndices,
        result
      )

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"SummarizerDAL compute took ${durationCompute} secs")

      val ret = if (OneCCL.isRoot()) {

        val convResultStartTime = System.nanoTime()
        val meanMatrix = OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.meanNumericTable))
        val varianceMatrix = OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.varianceNumericTable))
        val maxMatrix= OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.maximumNumericTable))
        val minMatrix = OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.minimumNumericTable))

        val meanVector = OldVectors.dense(meanMatrix.toArray)
        val varianceVector = OldVectors.dense(varianceMatrix.toArray)
        val maxVector = OldVectors.dense(maxMatrix.toArray)
        val minVector = OldVectors.dense(minMatrix.toArray)

        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"SummarizerDAL result conversion took ${durationCovResult} secs")

        Iterator((meanVector, varianceVector, maxVector, minVector))
      } else {
        Iterator.empty
      }

      OneCCL.cleanup()

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val meanVector = results(0)._1
    val varianceVector = results(0)._2
    val maxVector = results(0)._3
    val minVector = results(0)._4

    val summary = new MultivariateStatisticalDALSummary(meanVector, varianceVector, maxVector, minVector)

    summary
  }


  @native private def cSummarizerTrainDAL(data: Long,
                                           executor_num: Int,
                                           executor_cores: Int,
                                           useGPU: Boolean,
                                           gpuIndices: Array[Int],
                                           result: SummarizerResult): Long
}