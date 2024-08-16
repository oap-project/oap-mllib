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
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import com.intel.oneapi.dal.table.Common
import org.apache.spark.SparkException
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.regression.{LinearRegressionModel => MLlibLinearRegressionModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.RDD


/**
 * Model fitted by [[LinearRegressionDALImpl]].
 *
 * @param coefficients     model coefficients
 * @param intercept        model intercept
 * @param diagInvAtWA      diagonal of matrix (A^T * W * A)^-1
 * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
 */
// LinearRegressionDALModel is the same with WeightedLeastSquaresModel
// diagInvAtWA and objectiveHistory are not supported right now

private[mllib] class LinearRegressionDALModel(val coefficients: DenseVector,
                                              val intercept: Double,
                                              val diagInvAtWA: DenseVector,
                                              val objectiveHistory: Array[Double])
  extends Serializable {

  }

class LinearRegressionDALImpl( val fitIntercept: Boolean,
                               val regParam: Double,
                               val elasticNetParam: Double,
                               val standardizeFeatures: Boolean,
                               val standardizeLabel: Boolean,
                               val executorNum: Int,
                               val executorCores: Int)
  extends Serializable with Logging {

  require(regParam >= 0.0, s"regParam cannot be negative: $regParam")
  require(elasticNetParam >= 0.0 && elasticNetParam <= 1.0,
  s"elasticNetParam must be in [0, 1]: $elasticNetParam")

  /**
    * Creates a [[LinearRegressionDALModel]] from an RDD of [[Vector]]s.
    */

  def train(labeledPoints: Dataset[_],
            labelCol: String,
            featuresCol: String): LinearRegressionDALModel = {

    val sparkContext = labeledPoints.sparkSession.sparkContext
    val lrTimer = new Utils.AlgoTimeMetrics("LinearRegression", sparkContext)
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)

    val isTest = sparkContext.getConf.getBoolean("spark.oap.mllib.isTest", false)

    val kvsIPPort = getOneCCLIPPort(labeledPoints.rdd)
    lrTimer.record("Preprocessing")

    val labeledPointsTables = if (useDevice == "GPU") {
        if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
          OneDAL.coalesceLabelPointsToHomogenTables(labeledPoints,
            labelCol, featuresCol, executorNum, computeDevice)
        } else {
          val msg = s"OAP MLlib: Sparse table is not supported for GPU now."
          logError(msg)
          throw new SparkException(msg)
        }
    } else {
        if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
        OneDAL.coalesceLabelPointsToNumericTables(labeledPoints, labelCol, featuresCol, executorNum)
      } else {
        OneDAL.coalesceSparseLabelPointsToSparseNumericTables(labeledPoints,
          labelCol, featuresCol, executorNum)
      }
    }

    // OAP MLlib: Only normal linear regression is supported for GPU currently
    if (useDevice == "GPU" && regParam != 0) {
      val msg = s"OAP MLlib: Regularization parameter is not supported for GPU now."
      logError(msg)
      throw new SparkException(msg)
    }
    lrTimer.record("Data Convertion")

    labeledPointsTables.mapPartitionsWithIndex { (rank, iter) =>
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      OneCCL.setExecutorEnv("ZE_AFFINITY_MASK", gpuIndices(0).toString())
      Iterator.empty
    }.count()

    val results = labeledPointsTables.mapPartitionsWithIndex { (rank, tables) =>
        val (feature, label) = tables.next()
        val (featureTabAddr : Long, featureRows : Long, featureColumns : Long) =
          if (useDevice == "GPU") {
            feature
          } else {
            (feature.toString.toLong, 0L, 0L)
          }
        val (labelTabAddr : Long, labelRows : Long, labelColumns : Long) =
          if (useDevice == "GPU") {
            label
          } else {
            (label.toString.toLong, 0L, 0L)
          }

        OneCCL.init(executorNum, rank, kvsIPPort)
        val result = new LiRResult()

        val gpuIndices = if (useDevice == "GPU") {
          // OAP MLlib: This ia a hack for unit test, and will be repaireid in the future.
          // GPU info can't be detected automatically in UT enviroment.
          if (isTest) {
            Array(0)
          } else {
            val resources = TaskContext.get().resources()
            resources("gpu").addresses.map(_.toInt)
          }
        } else {
          null
        }

        val cbeta = cLinearRegressionTrainDAL(
          rank,
          featureTabAddr,
          featureRows,
          featureColumns,
          labelTabAddr,
          labelColumns,
          fitIntercept,
          regParam,
          elasticNetParam,
          executorNum,
          executorCores,
          computeDevice.ordinal(),
          gpuIndices,
          result
        )

        val ret = if (rank == 0) {
          val coefficientArray = if (useDevice == "GPU") {
              OneDAL.homogenTableToVectors(OneDAL.makeHomogenTable(cbeta),
                computeDevice)
            } else {
              OneDAL.numericTableToVectors(OneDAL.makeNumericTable(cbeta))
            }
          Iterator(coefficientArray(0))
        } else {
          Iterator.empty
        }
        OneCCL.cleanup()
        ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val coefficientVector = results(0)
    lrTimer.record("Training")
    lrTimer.print()

    val parentModel = new LinearRegressionDALModel(
      new DenseVector(coefficientVector.toArray.slice(1, coefficientVector.size)),
      coefficientVector(0), new DenseVector(Array(0D)), Array(0D))

    parentModel
  }

  // Single entry to call Linear Regression DAL backend with parameters
  @native private def cLinearRegressionTrainDAL(rank: Int,
                                  data: Long,
                                  numRows: Long,
                                  numCols: Long,
                                  label: Long,
                                  labelNumCols: Long,
                                  fitIntercept: Boolean,
                                  regParam: Double,
                                  elasticNetParam: Double,
                                  executorNum: Int,
                                  executorCores: Int,
                                  computeDeviceOrdinal: Int,
                                  gpuIndices: Array[Int],
                                  result: LiRResult): Long

  }
