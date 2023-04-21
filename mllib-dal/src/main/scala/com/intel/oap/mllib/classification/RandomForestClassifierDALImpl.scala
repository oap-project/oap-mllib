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
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common

import org.apache.spark.annotation.Since
import org.apache.spark.TaskContext
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
                                    val bootstrap: Boolean) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): (Matrix, Matrix,
                                   util.Map[Integer, util.ArrayList[LearningNode]]) = {
    logInfo(s"RandomForestClassifierDALImpl executorNum : " + executorNum)
    val sparkContext = labeledPoints.rdd.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val labeledPointsTables = if (useDevice == "GPU") {
      if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
        OneDAL.rddLabeledPointToMergedHomogenTables(labeledPoints,
          labelCol, featuresCol, executorNum, computeDevice)
      } else {
        OneDAL.rddLabeledPointToSparseCSRTables(labeledPoints,
          labelCol, featuresCol, executorNum, computeDevice)
      }
    } else {
      throw new Exception("Random Forset didn't implemente for CPU device, " +
        "Please run on GPU device.")
    }
    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)

    val results = labeledPointsTables.mapPartitionsWithIndex {
      (rank: Int, tables: Iterator[(Long, Long)]) =>
      val (featureTabAddr, lableTabAddr) = tables.next()

      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      OneCCL.init(executorNum, rank, kvsIPPort)
      val computeStartTime = System.nanoTime()
      val result = new RandomForestResult
      val hashmap = cRFClassifierTrainDAL(
        featureTabAddr,
        lableTabAddr,
        executorNum,
        computeDevice.ordinal(),
        classCount,
        treeCount,
        minObservationsLeafNode,
        minObservationsSplitNode,
        minWeightFractionLeafNode,
        minImpurityDecreaseSplitNode,
        bootstrap,
        gpuIndices,
        result)
      for ( (k, v) <- hashmap) {
        logInfo(s"key: $k, value: $v")
        for ( l <- v) {
          logInfo(l.toString)
        }
      }
      val computeEndTime = System.nanoTime()

      val durationCompute = (computeEndTime - computeStartTime).toDouble / 1E9

      logInfo(s"RandomForestClassifierDAL compute took ${durationCompute} secs")

      val ret = if (rank == 0) {
        val convResultStartTime = System.nanoTime()
        val probabilitiesNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.probabilitiesNumericTable),
          computeDevice)
        val predictionNumericTable = OneDAL.homogenTableToMatrix(
          OneDAL.makeHomogenTable(result.predictionNumericTable),
          computeDevice)
        val convResultEndTime = System.nanoTime()

        val durationCovResult = (convResultEndTime - convResultStartTime).toDouble / 1E9

        logInfo(s"RandomForestClassifierDAL result conversion took ${durationCovResult} secs")

        Iterator((probabilitiesNumericTable, predictionNumericTable, hashmap))
      } else {
        Iterator.empty
      }

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val hashmap = results(0)._3
    val myScalaMap = hashmap.mapValues(_.toSet)
    for ( (k, v) <- myScalaMap) {
      println(s"key: $k, value: $v")
      for ( l <- v) {
           println(l.toString)
      }
    }

    (results(0)._1, results(0)._2, results(0)._3)
  }

  @native private[mllib] def cRFClassifierTrainDAL(featureData: Long,
                                                   lableTabData: Long,
                                                   executorNum: Int,
                                                   computeDeviceOrdinal: Int,
                                                   classCount: Int,
                                                   treeCount: Int,
                                                   minObservationsLeafNode: Int,
                                                   minObservationsSplitNode: Int,
                                                   minWeightFractionLeafNode: Double,
                                                   minImpurityDecreaseSplitNode: Double,
                                                   bootstrap: Boolean,
                                                   gpuIndices: Array[Int],
                                                   result: RandomForestResult):
  java.util.HashMap[java.lang.Integer, java.util.ArrayList[LearningNode]]
}
