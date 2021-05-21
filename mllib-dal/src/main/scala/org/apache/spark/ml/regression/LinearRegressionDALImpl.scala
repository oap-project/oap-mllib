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

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
// import org.apache.spark.ml.linalg._
import org.apache.spark.ml.linalg.{Matrices, Matrix, Vector, Vectors, DenseVector, BLAS}
import org.apache.spark.ml.util.Utils
import org.apache.spark.ml.util.Utils.getOneCCLIPPort
import org.apache.spark.ml.util.{Instrumentation, OneCCL, OneDAL, Service}
import org.apache.spark.rdd.RDD

/**
 * Model fitted by [[LinearRegressionDALImpl]].
 *
 * @param coefficients model coefficients
 * @param intercept model intercept
 * @param diagInvAtWA diagonal of matrix (A^T * W * A)^-1
 * @param objectiveHistory objective function (scaled loss + regularization) at each iteration.
 */
// TODO: diagInvAtWA and objectiveHistory are not supported, keep them for compatibility?
private[ml] class LRDALModel(
    val coefficients: DenseVector,
    val intercept: Double,
    val diagInvAtWA: DenseVector,
    val objectiveHistory: Array[Double]) extends Serializable {

  // def predict(features: Vector): Double = {
  //   BLAS.dot(coefficients, features) + intercept
  // }
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
  extends Serializable with Logging {

  require(regParam >= 0.0, s"regParam cannot be negative: $regParam")
  require(elasticNetParam >= 0.0 && elasticNetParam <= 1.0,
    s"elasticNetParam must be in [0, 1]: $elasticNetParam")
  require(maxIter > 0, s"maxIter must be a positive integer: $maxIter")
  require(tol >= 0.0, s"tol must be >= 0, but was set to $tol")

  /**
   * Creates a [[LRDALModel]] from an RDD of [[Vector]]s.
   */
  def fitWithDAL(labeledPoints: RDD[(Vector, Double)],
            instr: Option[Instrumentation]): LRDALModel = {
    // if (regParam == 0.0) {
    //   instr.logWarning("regParam is zero, which might cause numerical instability and overfitting.")
    // }

/*****************************************************************/
    // val coalescedTables = OneDAL.rddLabeledPointToMergedTables(labeledPoints, executorNum)

    // val executorIPAddress = Utils.sparkFirstExecutorIP(coalescedTables.sparkContext)
    // val kvsIP = coalescedTables.sparkContext.conf.get("spark.oap.mllib.oneccl.kvs.ip",
    //   executorIPAddress)
    // val kvsPortDetected = Utils.checkExecutorAvailPort(coalescedTables, kvsIP)
    // println(s"\nkvsPortDetected: ${kvsPortDetected}")
    // val kvsPort = coalescedTables.sparkContext.conf.getInt("spark.oap.mllib.oneccl.kvs.port",
    //   kvsPortDetected)

    // val kvsIPPort = kvsIP + "_" + kvsPort
    // println(s"\nkvsIPPort: ${kvsIPPort}")

    // TODO: use getOneCCLIPPort() to replace the above code
    // val kvsIPPort = getOneCCLIPPort(labeledPoints)
    val kvsIPPort = "10.1.2.142_3000"

    val labeledPointsTables = OneDAL.rddLabeledPointToMergedTables(labeledPoints, executorNum)

    val results = labeledPointsTables.mapPartitionsWithIndex {
      case (rank: Int, tables: Iterator[(Long, Long)]) =>
        val (featureTabAddr, lableTabAddr) = tables.next()

        OneCCL.init(executorNum, rank, kvsIPPort)

        val result = new LiRResult
        cLRTrainDAL(
          featureTabAddr,
          lableTabAddr,
          fitIntercept,
          regParam,
          elasticNetParam,
          executorNum,
          executorCores,
          result
        )

        val ret = if (OneCCL.isRoot()) {
          val coefficientArray = OneDAL.numericTableToVectors(OneDAL.makeNumericTable(result.coeffNumericTable))
          // println(s"\ncoefficientArray: ${coefficientArray}, size: ${coefficientArray.size}")
          Iterator((coefficientArray, result.intercept))
        } else {
          Iterator.empty
        }

        OneCCL.cleanup()
        ret
    }.collect()
    println(s"\nresults.length: ${results.length}")
    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val coefficientArray = results(0)._1  // TODO: get coefficients from results
    val intercept = results(0)._2     // TODO: get intercept from results

    // println(s"\ncoefficientArray: ${coefficientArray}, size: ${coefficientArray(0).toDense.size}")
    println(s"\nintercept: ${intercept}")

    val parentModel = new LRDALModel(coefficientArray(0).toDense, intercept, new DenseVector(Array(0D)), Array(0D))
    println(s"\nnew LRDALModel() done.\n")

    parentModel
  }

  // Single entry to call Linear Regression DAL backend with parameters
  @native private def cLRTrainDAL(data: Long,
                                  label: Long,
                                  fitIntercept: Boolean,
                                  regParam: Double,
                                  elasticNetParam: Double,
                                  executor_num: Int,
                                  executor_cores: Int,
                                  result: LiRResult): Long

}
