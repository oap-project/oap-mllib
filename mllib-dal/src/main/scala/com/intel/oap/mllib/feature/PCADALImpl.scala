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

package com.intel.oap.mllib.feature

import com.intel.daal.data_management.data.{HomogenNumericTable, NumericTable}
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL}
import org.apache.spark.TaskContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.feature.{PCAModel => MLlibPCAModel, StandardScaler => MLlibStandardScaler}
import org.apache.spark.mllib.linalg.{DenseVector => OldDenseVector, DenseMatrix => OldDenseMatrix, Vectors => OldVectors}
import org.apache.spark.rdd.RDD

import java.util.Arrays

class PCADALModel private[mllib] (
  val k: Int,
  val pc: OldDenseMatrix,
  val explainedVariance: OldDenseVector)

class PCADALImpl(val k: Int,
                 val executorNum: Int,
                 val executorCores: Int)
  extends Serializable with Logging {

  def train(data: RDD[Vector]): PCADALModel = {

    val normalizedData = normalizeData(data)

    val coalescedTables = OneDAL.rddVectorToMergedTables(normalizedData, executorNum)

    val kvsIPPort = getOneCCLIPPort(coalescedTables)

    val sparkContext = data.sparkContext
    val useGPU = sparkContext.getConf.getBoolean("spark.oap.mllib.useGPU", false)

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      val gpuIndices = if (useGPU) {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }

      val tableArr = table.next()
      OneCCL.init(executorNum, rank, kvsIPPort)

      val result = new PCAResult()
      cPCATrainDAL(
        tableArr,
        k,
        executorNum,
        executorCores,
        useGPU,
        gpuIndices,
        result
      )

      val ret = if (OneCCL.isRoot()) {

        val pcNumericTable = OneDAL.makeNumericTable(result.pcNumericTable)
        val explainedVarianceNumericTable = OneDAL.makeNumericTable(
          result.explainedVarianceNumericTable)

        val principleComponents = getPrincipleComponentsFromDAL(pcNumericTable, k)
        val explainedVariance = getExplainedVarianceFromDAL(explainedVarianceNumericTable, k)

        Iterator((principleComponents, explainedVariance))
      } else {
        Iterator.empty
      }

      OneCCL.cleanup()

      ret
    }.collect()

    // Make sure there is only one result from rank 0
    assert(results.length == 1)

    val pc = results(0)._1
    val explainedVariance = results(0)._2

    val parentModel = new PCADALModel(k,
      OldDenseMatrix.fromML(pc),
      OldVectors.fromML(explainedVariance).toDense
    )

    parentModel
  }

  // Normalize data before training
  private def normalizeData(input: RDD[Vector]): RDD[Vector] = {
    val vectors = input.map(OldVectors.fromML(_))
    val scaler = new MLlibStandardScaler(withMean = true, withStd = false).fit(vectors)
    val res = scaler.transform(vectors)
    res.map(_.asML)
  }

  private def getPrincipleComponentsFromDAL(table: NumericTable, k: Int): DenseMatrix = {
    val data = table.asInstanceOf[HomogenNumericTable].getDoubleArray()

    val numRows = table.getNumberOfRows.toInt

    require(k <= numRows, "k should be less or equal to row number")

    val numCols = table.getNumberOfColumns.toInt

    // Column-major, transpose of top K rows of NumericTable
    new DenseMatrix(numCols, k, data.slice(0, numCols * k), false)

//    val result = DenseMatrix.zeros(numCols, k)
//
//    for (row <- 0 until k) {
//      for (col <- 0 until numCols) {
//        result(col, row) = data(row * numCols + col)
//      }
//    }
//
//    result
  }

  private def getExplainedVarianceFromDAL(table_1xn: NumericTable, k: Int): DenseVector = {
    val data = table_1xn.asInstanceOf[HomogenNumericTable].getDoubleArray()
    val sum = data.sum
    val topK = Arrays.copyOfRange(data, 0, k)
    for (i <- 0 until k)
      topK(i) = topK(i) / sum
    new DenseVector(topK)
  }

  // Single entry to call Correlation PCA DAL backend with parameter K
  @native private def cPCATrainDAL(data: Long,
                                   k: Int,
                                   executor_num: Int,
                                   executor_cores: Int,
                                   useGPU: Boolean,
                                   gpuIndices: Array[Int],
                                   result: PCAResult): Long

}
