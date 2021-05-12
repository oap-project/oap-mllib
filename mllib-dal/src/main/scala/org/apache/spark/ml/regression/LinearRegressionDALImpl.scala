/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import java.util.Arrays
import com.intel.daal.data_management.data.{HomogenNumericTable, NumericTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.util.{OneCCL, OneDAL, Utils}
// import org.apache.spark.mllib.feature.{PCAModel => MLlibPCAModel}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.{StandardScaler => MLlibStandardScaler}

/**
 * Model fitted by [[LinearRegressionDALImpl]].
 *
 * @param coefficients model coefficients
 * @param intercept model intercept
 * @param diagInvAtWA diagonal of matrix (A^T * W * A)^-1
 * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
 */
// TODO: diagInvAtWA and objectiveHistory are not supported, keep them for compatibility
private[ml] class LRDALModel(
    val coefficients: DenseVector,
    val intercept: Double,
    val diagInvAtWA: DenseVector,
    val objectiveHistory: Array[Double]) extends Serializable {

// TODO: does predict() need update?
  def predict(features: Vector): Double = {
    BLAS.dot(coefficients, features) + intercept
  }
}

class LinearRegressionDALImpl (
    val fitIntercept: Boolean,
    val regParam: Double,
    val elasticNetParam: Double,
    val standardizeFeatures: Boolean,
    val standardizeLabel: Boolean,
    val maxIter: Int = 100,
    val tol: Double = 1e-6,
    val executorNum: Int,
    val executorCores: Int)
  // extends Serializable with Logging {
  {

  require(regParam >= 0.0, s"regParam cannot be negative: $regParam")
  require(elasticNetParam >= 0.0 && elasticNetParam <= 1.0,
    s"elasticNetParam must be in [0, 1]: $elasticNetParam")
  require(maxIter > 0, s"maxIter must be a positive integer: $maxIter")
  require(tol >= 0.0, s"tol must be >= 0, but was set to $tol")

  /**
   * Creates a [[LRDALModel]] from an RDD of [[Vector]]s.
   */
  def fitWithDAL(data: RDD[Vector]): LRDALModel = {
    // if (regParam == 0.0) {
    //   instr.logWarning("regParam is zero, which might cause numerical instability and overfitting.")
    // }

/*****************************************************************/
    val coalescedTables = OneDAL.vectorsToMergedNumericTables(data, executorNum)
    val executorIPAddress = Utils.sparkFirstExecutorIP(coalescedTables.sparkContext)
    val kvsIP = coalescedTables.sparkContext.conf.get("spark.oap.mllib.oneccl.kvs.ip", executorIPAddress)

    val kvsPortDetected = Utils.checkExecutorAvailPort(coalescedTables, kvsIP)
    val kvsPort = coalescedTables.sparkContext.conf.getInt("spark.oap.mllib.oneccl.kvs.port", kvsPortDetected)

    val kvsIPPort = kvsIP+"_"+kvsPort

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      val tableArr = table.next()
      OneCCL.init(executorNum, rank, kvsIPPort)

      val result = new LiRResult()
      cLRTrainDAL(
        tableArr,
        fitIntercept,
        regParam,
        elasticNetParam,
        executorNum,
        executorCores,
        result
      )

      val ret = if (OneCCL.isRoot()) {
        val coefficientArray = OneDAL.numericTableToVectors(OneDAL.makeNumericTable(result.coeffNumericTable))
        Iterator((coefficientArray, result.intercept))
      } else {
        Iterator.empty
      }

      OneCCL.cleanup()

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val coefficientArray = results(0)._1  // TODO: get coefficients from results
    val intercept = results(0)._2     // TODO: get intercept from results

    val parentModel = new LRDALModel(coefficientArray(0).toDense, intercept, new DenseVector(Array(0D)), Array(0D))

    parentModel
  }

  private def getPrincipleComponentsFromDAL(table: NumericTable, k: Int): DenseMatrix = {
    val data = table.asInstanceOf[HomogenNumericTable].getDoubleArray()

    val numRows = table.getNumberOfRows.toInt

    require(k <= numRows, "k should be less or equal to row number")

    val numCols = table.getNumberOfColumns.toInt

    val result = DenseMatrix.zeros(numCols, k)

    for (row <- 0 until k) {
      for (col <- 0 until numCols) {
        result(col, row) = data(row * numCols + col)
      }
    }

    result
  }

  private def getExplainedVarianceFromDAL(table_1xn: NumericTable, k: Int): DenseVector = {
    val data = table_1xn.asInstanceOf[HomogenNumericTable].getDoubleArray()
    val sum = data.sum
    val topK = Arrays.copyOfRange(data, 0, k)
    for ( i <- 0 until k )
      topK(i) = topK(i) / sum
    new DenseVector(topK)
  }

  // Single entry to call Linear Regression DAL backend with parameters
  @native private def cLRTrainDAL(data: Long,
                                  fitIntercept: Boolean,
                                  regParam: Double,
                                  elasticNetParam: Double,
                                  executor_num: Int,
                                  executor_cores: Int,
                                  result: LiRResult): Long

}
