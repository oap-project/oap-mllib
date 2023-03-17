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
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{DenseVector, Vector}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.regression.{LinearRegressionModel => MLlibLinearRegressionModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.sql.Dataset
import org.apache.spark.rdd.RDD

class LRDALImpl(val fitIntercept: Boolean,
  val regParam: Double,
  val elasticNetParam: Double,
  val standardizeFeatures: Boolean,
  val standardizeLabel: Boolean,
  val executorNum: Int,
  val executorCores: Int ) extends Serializable with Logging {

  def train(labeledPoints: Dataset[_],
              labelCol: String,
              featuresCol: String): MLlibLinearRegressionModel = {

    val labeledPointsTables = if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
      OneDAL.rddLabeledPointToMergedTables(labeledPoints, labelCol, featuresCol, executorNum)
    } else {
      OneDAL.rddLabeledPointToSparseTables(labeledPoints, labelCol, featuresCol, executorNum)
    }

    val sparkContext = labeledPointsTables.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)

    val kvsIPPort = getOneCCLIPPort(labeledPointsTables)
    val results = labeledPointsTables.mapPartitionsWithIndex { case (rank: Int, tables: Iterator[(Long, Long)]) =>
      val tableArr = tables.next()
      val (featureTabAddr, lableTabAddr) = tables.next()
      //create an result and send to implement
      val result = new LiRResult()
      OneCCL.initDpcpp()

      val coeffNumericTable = cLinearRegressionGPUTrainDAL(
        featureTabAddr,
        lableTabAddr,
        regParam,
        elasticNetParam,
        executorNum,
        computeDevice.ordinal(),
        rank,
        kvsIPPort,
        result
      )
      val ret = if (rank == 0) {
          val coefficientArray = OneDAL.homogenTableToVectors(OneDAL.makeHomogenTable(coeffNumericTable),
            computeDevice)
          Iterator(coefficientArray)
        } else {
          Iterator.empty
        }
      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)
    val coefficientVector = results(0)

    val coefficientResult = coefficientVector.map(OldVectors.fromML(_)).slice(1, coefficientVector.size)
    val interceptResult = coefficientVector.map(OldVectors.fromML(_))
    val parentModel = new MLlibLinearRegressionModel(
      coefficientResult(0),
      interceptResult(0)(0))

    parentModel
  }

  @native private[mllib] def cLinearRegressionGPUTrainDAL(data: Long,
                                            label: Long,
                                            regParam: Double,
                                            elasticNetParam: Double,
                                            executorNum: Int,
                                            computeDeviceOrdinal: Int,
                                            rankId: Int,
                                            ipPort: String,
                                            result: LiRResult): Long

  }


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

      val kvsIPPort = getOneCCLIPPort(labeledPoints.rdd)

      val labeledPointsTables = if (OneDAL.isDenseDataset(labeledPoints, featuresCol)) {
        OneDAL.rddLabeledPointToMergedTables(labeledPoints, labelCol, featuresCol, executorNum)
      } else {
        OneDAL.rddLabeledPointToSparseTables(labeledPoints, labelCol, featuresCol, executorNum)
      }

      val results = labeledPointsTables.mapPartitionsWithIndex {
        case (rank: Int, tables: Iterator[(Long, Long)]) =>
          val (featureTabAddr, lableTabAddr) = tables.next()

          OneCCL.init(executorNum, rank, kvsIPPort)

          val result = new LiRResult
          cLinearRegressionTrainDAL(
            featureTabAddr,
            lableTabAddr,
            regParam,
            elasticNetParam,
            executorNum,
            executorCores,
            result
          )

          val ret = if (OneCCL.isRoot()) {
            val coefficientArray = OneDAL.numericTableToVectors(
              OneDAL.makeNumericTable(result.coeffNumericTable))
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

      val parentModel = new LinearRegressionDALModel(
        new DenseVector(coefficientVector.toArray.slice(1, coefficientVector.size)),
        coefficientVector(0), new DenseVector(Array(0D)), Array(0D))

      parentModel
    }

    // Single entry to call Linear Regression DAL backend with parameters
    @native private def cLinearRegressionTrainDAL(data: Long,
                                    label: Long,
                                    regParam: Double,
                                    elasticNetParam: Double,
                                    executor_num: Int,
                                    executor_cores: Int,
                                    result: LiRResult): Long

  }
