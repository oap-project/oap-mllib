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

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.rdd.RDD

class CorrelationDALImpl(
                          val executorNum: Int,
                          val executorCores: Int)
  extends Serializable with Logging {

  def computeCorrelationMatrix(data: RDD[Vector]): Matrix = {
    val sparkContext = data.sparkContext
    val corTimer = new Utils.AlgoTimeMetrics("Correlation", sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    corTimer.record("Preprocessing")

    val coalescedTables = if (useDevice == "GPU") {
      OneDAL.coalesceVectorsToHomogenTables(data, executorNum,
        computeDevice)
    } else {
      OneDAL.coalesceVectorsToNumericTables(data, executorNum)
    }
    corTimer.record("Data Convertion")

    val kvsIPPort = getOneCCLIPPort(coalescedTables)

    coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      OneCCL.init(executorNum, rank, kvsIPPort, computeDevice.ordinal())
      Iterator.empty
    }.count()
    corTimer.record("OneCCL Init")

    val results = coalescedTables.mapPartitionsWithIndex { (rank, iter) =>
      val (tableArr : Long, rows : Long, columns : Long) = if (useDevice == "GPU") {
        iter.next()
      } else {
        (iter.next().toString.toLong, 0L, 0L)
      }

      val computeStartTime = System.nanoTime()

      val result = new CorrelationResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      cCorrelationTrainDAL(
        rank,
        tableArr,
        rows,
        columns,
        executorNum,
        executorCores,
        computeDevice.ordinal(),
        gpuIndices,
        result
      )

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"CorrelationDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val correlationNumericTable = if (useDevice == "GPU") {
          OneDAL.homogenTableToMatrix(OneDAL.makeHomogenTable(result.getCorrelationNumericTable),
            computeDevice)
        } else {
          OneDAL.numericTableToMatrix(OneDAL.makeNumericTable(result.getCorrelationNumericTable))
        }
        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"CorrelationDAL result conversion took ${durationCovResult} secs")

        Iterator(correlationNumericTable)
      } else {
        Iterator.empty
      }
      OneCCL.cleanup()
      ret
    }.collect()
    corTimer.record("Training")
    corTimer.print()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val correlationMatrix = results(0)

    correlationMatrix
  }


  @native private[mllib] def cCorrelationTrainDAL(rank: Int,
                                           data: Long,
                                           numRows: Long,
                                           numCols: Long,
                                           executorNum: Int,
                                           executorCores: Int,
                                           computeDeviceOrdinal: Int,
                                           gpuIndices: Array[Int],
                                           result: CorrelationResult): Long
}
