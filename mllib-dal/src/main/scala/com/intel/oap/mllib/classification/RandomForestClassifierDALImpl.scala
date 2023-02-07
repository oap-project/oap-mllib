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
import com.intel.oap.mllib.stat.CorrelationResult
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.Node
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class DecisionTreeClassificationModel private[mllib] (val rootNode: Node,
                                                    val numFeatures: Int,
                                                    val numClasses: Int)

private[mllib] class RandomForestClassificationModel (
                         val _trees: Array[DecisionTreeClassificationModel],
                         val numFeatures: Int,
                         val numClasses: Int)

class RandomForestClassifierDALImpl(val classCount: Int,
                                    val treeCount: Int,
                                    val featurePerNode: Int,
                                    val minObservationsLeafNode: Int,
                                    val minObservationsSplitNode: Int,
                                    val minWeightFractionLeafNode: Double,
                                    val minImpurityDecreaseSplitNode: Double,
                                    val executorNum: Int,
                                    val executorCores: Int) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): RandomForestClassificationModel = {
    val sparkContext = labeledPoints.rdd.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val labeledPointsTables = if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
      OneDAL.rddLabeledPointToMergedHomogenTables(labeledPoints,
        labelCol, featuresCol, executorNum, computeDevice)
    } else {
      OneDAL.rddLabeledPointToSparseCSRTables(labeledPoints,
        labelCol, featuresCol, executorNum, computeDevice)
    }
    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)

    val results = labeledPointsTables.mapPartitionsWithIndex {
      (rank: Int, tables: Iterator[(Long, Long)]) =>
      val (featureTabAddr, lableTabAddr) = tables.next()

      OneCCL.initDpcpp()

      val computeStartTime = System.nanoTime()

      val result = new RandomForestResult()
      cRFClassifierTrainDAL(
        featureTabAddr,
        lableTabAddr,
        executorNum,
        computeDevice.ordinal(),
        rank,
        classCount,
        treeCount,
        minObservationsLeafNode,
        minObservationsSplitNode,
        minWeightFractionLeafNode,
        minImpurityDecreaseSplitNode,
        kvsIPPort,
        result
      )

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"RandomForestClassifierDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val probabilitiesNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.probabilitiesNumericTable),
          computeDevice)

        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"RandomForestClassifierDAL result conversion took ${durationCovResult} secs")

        Iterator(probabilitiesNumericTable)
      } else {
        Iterator.empty
      }

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    val parentModel = new RandomForestClassificationModel(null, 0, 0)
    parentModel

  }

  @native private[mllib] def cRFClassifierTrainDAL(featureTabAddr: Long,
                                             lableTabAddr: Long,
                                             executorNum: Int,
                                             computeDeviceOrdinal: Int,
                                             classCount: Int,
                                             rankId: Int,
                                             treeCount: Int,
                                             minObservationsLeafNode: Int,
                                             minObservationsSplitNode: Int,
                                             minWeightFractionLeafNode: Double,
                                             minImpurityDecreaseSplitNode: Double,
                                             ipPort: String,
                                             result: RandomForestResult): Long
}
