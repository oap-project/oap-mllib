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
package com.intel.oap.mllib.regression

import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.classification.{LearningNode, RandomForestResult}
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.Dataset

import java.util
import scala.collection.JavaConversions._

class RandomForestRegressorDALImpl(val uid: String,
                                    val treeCount: Int,
                                    val featurePerNode: Int,
                                    val minObservationsLeafNode: Int,
                                    val minObservationsSplitNode: Int,
                                    val minWeightFractionLeafNode: Double,
                                    val minImpurityDecreaseSplitNode: Double,
                                    val executorNum: Int,
                                    val executorCores: Int,
                                    val maxTreeDepth: Int,
                                    val seed: Long,
                                    val maxbins: Int,
                                    val bootstrap: Boolean) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): (util.Map[Integer, util.ArrayList[LearningNode]]) = {
    logInfo(s"RandomForestRegressorDALImpl executorNum : " + executorNum)
    val sparkContext = labeledPoints.rdd.sparkContext
    val rfrTimer = new Utils.AlgoTimeMetrics("RandomForestRegressor", sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    // used run Random Forest unit test
    val isTest = sparkContext.getConf.getBoolean("spark.oap.mllib.isTest", false)
    rfrTimer.record("Preprocessing")

    val labeledPointsTables = if (useDevice == "GPU") {
      if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
        OneDAL.coalesceLabelPointsToHomogenTables(labeledPoints,
          labelCol, featuresCol, executorNum, computeDevice)
      } else {
        throw new Exception("Oneapi didn't implement sparse dataset")
      }
    } else {
      throw new Exception("Random Forset didn't implemente for CPU device, " +
        "Please run on GPU device.")
    }
    rfrTimer.record("Data Convertion")

    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)

    labeledPointsTables.mapPartitionsWithIndex { (rank, table) =>
      OneCCL.init(executorNum, rank, kvsIPPort)
      Iterator.empty
    }.count()
    rfrTimer.record("OneCCL Init")

    val results = labeledPointsTables.mapPartitionsWithIndex {
      (rank: Int, tables: Iterator[(String, String)]) =>
      val (feature, label) = tables.next()
      val (featureTabAddr : Long, featureRows : Long, featureColumns : Long) =
        if (useDevice == "GPU") {
          val parts = feature.toString.split("_")
          (parts(0).toLong, parts(1).toLong, parts(2).toLong)
        } else {
          (feature.toString.toLong, 0, 0)
        }
      val (labelTabAddr : Long, labelRows : Long, labelColumns : Long) =
        if (useDevice == "GPU") {
          val parts = feature.toString.split("_")
          (parts(0).toLong, parts(1).toLong, parts(2).toLong)
        } else {
          (label.toString.toLong, 0, 0)
        }
      val gpuIndices = if (useDevice == "GPU") {
        if (isTest) {
          Array(0)
        } else {
          val resources = TaskContext.get().resources()
          resources("gpu").addresses.map(_.toInt)
        }
      } else {
        null
      }

      val computeStartTime = System.nanoTime()
      val result = new RandomForestResult
      val hashmap = cRFRegressorTrainDAL(
        featureTabAddr,
        featureRows,
        featureColumns,
        labelTabAddr,
        labelColumns,
        executorNum,
        computeDevice.ordinal(),
        treeCount,
        featurePerNode,
        minObservationsLeafNode,
        maxTreeDepth,
        seed,
        maxbins,
        bootstrap,
        gpuIndices,
        result)

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"RandomForestRegressorDALImpl compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val predictionNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.getPredictionNumericTable),
          computeDevice)
        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"RandomForestRegressorDALImpl result conversion took ${durationCovResult} secs")

        Iterator((predictionNumericTable, hashmap))
      } else {
        Iterator.empty
      }

      ret
    }.collect()
    rfrTimer.record("Training")
    rfrTimer.print()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    results(0)._2
  }

  @native private[mllib] def cRFRegressorTrainDAL(featureTabAddr: Long,
                                             numRows: Long,
                                             numCols: Long,
                                             lableTabAddr: Long,
                                             labelNumCols: Long,
                                             executorNum: Int,
                                             computeDeviceOrdinal: Int,
                                             treeCount: Int,
                                             featurePerNode: Int,
                                             minObservationsLeafNode: Int,
                                             maxTreeDepth: Int,
                                             seed: Long,
                                             maxbins: Int,
                                             bootstrap: Boolean,
                                             gpuIndices: Array[Int],
                                             result: RandomForestResult): java.util.HashMap[java.lang.Integer, java.util.ArrayList[LearningNode]]
}
