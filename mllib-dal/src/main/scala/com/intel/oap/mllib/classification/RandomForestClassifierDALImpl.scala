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
import com.intel.oap.mllib.{CommonJob, OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.annotation.Since
import org.apache.spark.{SparkEnv, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.tree.{InternalNode, LeafNode, Node, Split}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.tree
import org.apache.spark.mllib.tree.model.ImpurityStats

import java.util
import java.util.{ArrayList, Map}
import scala.collection.mutable.HashMap
import scala.collection.JavaConversions._

class RandomForestClassifierDALImpl(val uid: String,
                                    val classCount: Int,
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
                                    val maxBins: Int,
                                    val bootstrap: Boolean) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): (util.Map[Integer, util.ArrayList[LearningNode]]) = {

    logInfo(s"RandomForestClassifierDALImpl executorNum : " + executorNum)
    val sparkContext = labeledPoints.rdd.sparkContext
    val rfcTimer = new Utils.AlgoTimeMetrics("RandomForestClassifier", sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    // used run Random Forest unit test
    val isTest = sparkContext.getConf.getBoolean("spark.oap.mllib.isTest", false)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    rfcTimer.record("Preprocessing")
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
    rfcTimer.record("Data Convertion")
    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)

    CommonJob.initCCLAndSetAffinityMask(labeledPointsTables, executorNum, kvsIPPort, useDevice)
    rfcTimer.record("OneCCL Init")

    val results = labeledPointsTables.mapPartitionsWithIndex { (rank, tables) =>
      val (feature, label) = tables.next()
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
      val hashmap = cRFClassifierTrainDAL(
        rank,
        feature._1,
        feature._2,
        feature._3,
        label._1,
        label._3,
        executorNum,
        computeDevice.ordinal(),
        classCount,
        treeCount,
        featurePerNode,
        minObservationsLeafNode,
        minObservationsSplitNode,
        minWeightFractionLeafNode,
        minImpurityDecreaseSplitNode,
        maxTreeDepth,
        seed,
        maxBins,
        bootstrap,
        gpuIndices,
        result)

      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"RandomForestClassifierDAL compute took ${durationCompute} secs")

      val isRoot = if (useDevice == "GPU") {
        SparkEnv.get.executorId.toInt == 0
      } else {
        rank == 0
      }
      val ret = if (isRoot) {
        Iterator(hashmap)
      } else {
        Iterator.empty
      }
      OneCCL.cleanup()
      ret
    }.collect()

    rfcTimer.record("Training")
    rfcTimer.print()
    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    results(0)
  }

  @native private[mllib] def cRFClassifierTrainDAL(rank: Int,
                                                   featureTabAddr: Long,
                                                   numRows: Long,
                                                   numCols: Long,
                                                   lableTabAddr: Long,
                                                   labelNumCols: Long,
                                                   executorNum: Int,
                                                   computeDeviceOrdinal: Int,
                                                   classCount: Int,
                                                   treeCount: Int,
                                                   featurePerNode: Int,
                                                   minObservationsLeafNode: Int,
                                                   minObservationsSplitNode: Int,
                                                   minWeightFractionLeafNode: Double,
                                                   minImpurityDecreaseSplitNode: Double,
                                                   maxTreeDepth: Int,
                                                   seed: Long,
                                                   maxBins: Int,
                                                   bootstrap: Boolean,
                                                   gpuIndices: Array[Int],
                                                   result: RandomForestResult):
  java.util.HashMap[java.lang.Integer, java.util.ArrayList[LearningNode]]
}
