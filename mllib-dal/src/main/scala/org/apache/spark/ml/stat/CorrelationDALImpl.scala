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

package org.apache.spark.ml.stat

import java.util.Arrays

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.util.{OneCCL, OneDAL, Utils}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Utils.getOneCCLIPPort
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix}


class CorrelationDALImpl(
                 val executorNum: Int,
                 val executorCores: Int)
  extends Serializable with Logging {

    def computeCorrelationMatrix(data: RDD[Vector] ): Matrix = {

      val kvsIPPort = getOneCCLIPPort(data)
      println(s"OneCCL ip port : ${kvsIPPort} ")

      val coalescedTables = OneDAL.rddVectorToMergedTables(data, executorNum)
      println(s"executorNum : ${executorNum} ")
      println(s"executorCores : ${executorCores} ")

      val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
        println(s"map partition :")
        val tableArr = table.next()

        OneCCL.init(executorNum, rank, kvsIPPort)

        val computeStartTime = System.nanoTime()

        val result = new CorrelationResult()
        cCorrelationTrainDAL(
          tableArr,
          executorNum,
          executorCores,
          result
        )

        val computeEndTime = System.nanoTime()

        val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

        println(s"CorrelationDAL compute took ${durationCompute} secs")

        val ret = if (OneCCL.isRoot()) {

          val convResultStartTime = System.nanoTime()
          val covarianceNumericTable = OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.covarianceNumericTable))
          val correlationNumericTable = OneDAL.numericTableToOldMatrix(OneDAL.makeNumericTable(result.correlationNumericTable))
          val meanNumericTable = OneDAL.numericTableToVectors(OneDAL.makeNumericTable(result.meanNumericTable))

          val convResultEndTime = System.nanoTime()

          val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

          println(s"CorrelationDAL result conversion took ${durationCovResult} secs")

          Iterator((covarianceNumericTable, correlationNumericTable, meanNumericTable))
        } else {
          Iterator.empty
        }

        OneCCL.cleanup()

        ret
      }.collect()

      // Make sure there is only one result from rank 0
      assert(results.length == 1)

      // Release native memory for numeric tables
      OneDAL.releaseNumericTables(data.sparkContext)

      val covarianceMatrix = results(0)._1
      val correlationMatrix = results(0)._2
      val meanVectors = results(0)._3

      correlationMatrix
      }


  @native private def cCorrelationTrainDAL(data: Long,
                                   executor_num: Int,
                                   executor_cores: Int,
                                   result: CorrelationResult): Long

  }