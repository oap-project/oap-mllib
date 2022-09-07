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

import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalDALSummary, MultivariateStatisticalSummary => Summary}
import org.apache.spark.rdd.RDD
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oneapi.dal.table.Common

class SummarizerDALImpl(val executorNum: Int,
                        val executorCores: Int)
  extends Serializable with Logging {

  def computeSummarizerMatrix(data: RDD[Vector]): Summary = {
    val sparkContext = data.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val coalescedTables = OneDAL.rddVectorToMergedHomogenTables(data, executorNum, computeDevice)
    val kvsIPPort = getOneCCLIPPort(data)

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      val tableArr = table.next()
      OneCCL.init(executorNum, rank, kvsIPPort)

      val computeStartTime = System.nanoTime()

      val result = new SummarizerResult()
      cSummarizerTrainDAL(
        tableArr,
        executorNum,
        computeDevice.ordinal(),
        rank,
        kvsIPPort,
        result
      )

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"SummarizerDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {

        val convResultStartTime = System.nanoTime()
        val meanVector = OneDAL.homogenTable1xNToVector(
          OneDAL.makeHomogenTable(result.meanNumericTable), computeDevice)
        val varianceVector = OneDAL.homogenTable1xNToVector(
          OneDAL.makeHomogenTable(result.varianceNumericTable), computeDevice)
        val maxVector = OneDAL.homogenTable1xNToVector(
          OneDAL.makeHomogenTable(result.maximumNumericTable), computeDevice)
        val minVector = OneDAL.homogenTable1xNToVector(
          OneDAL.makeHomogenTable(result.minimumNumericTable), computeDevice)

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

    val summary = new MultivariateStatisticalDALSummary(OldVectors.fromML(meanVector),
                                                        OldVectors.fromML(varianceVector),
                                                        OldVectors.fromML(maxVector),
                                                        OldVectors.fromML(minVector))

    summary
  }

  @native private[mllib] def cSummarizerTrainDAL(data: Long,
                                          executor_num: Int,
                                          computeDeviceOrdinal: Int,
                                          rankId: Int,
                                          ipPort: String,
                                          result: SummarizerResult): Long
}
