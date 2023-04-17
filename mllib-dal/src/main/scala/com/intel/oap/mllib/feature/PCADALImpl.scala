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

import java.nio.DoubleBuffer
import com.intel.daal.data_management.data.{HomogenNumericTable, NumericTable}
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Service, Utils}
import org.apache.spark.TaskContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.feature.{PCAModel => MLlibPCAModel, StandardScaler => MLlibStandardScaler}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, DenseVector => OldDenseVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD

import java.util.Arrays
import com.intel.oneapi.dal.table.{Common, HomogenTable, RowAccessor}

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
    val sparkContext = normalizedData.sparkContext
    val useDevice = sparkContext.getConf.get("spark.oap.mllib.device", Utils.DefaultComputeDevice)
    val computeDevice = Common.ComputeDevice.getDeviceByName(useDevice)
    val coalescedTables = if (useDevice == "GPU") {
      OneDAL.coalesceToHomogenTables(normalizedData, executorNum,
        computeDevice)
    } else {
      OneDAL.rddVectorToMergedTables(normalizedData, executorNum)
    }
    val kvsIPPort = getOneCCLIPPort(coalescedTables)

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      val tableArr = table.next()
      OneCCL.init(executorNum, rank, kvsIPPort)
      val result = new PCAResult()
      val gpuIndices = if (useDevice == "GPU") {
        val resources = TaskContext.get().resources()
        resources("gpu").addresses.map(_.toInt)
      } else {
        null
      }
      cPCATrainDAL(
        tableArr,
        executorNum,
        executorCores,
        computeDevice.ordinal(),
        gpuIndices,
        result
      )

      val ret = if (rank == 0) {
        val principleComponents = if (useDevice == "GPU") {
          val pcNumericTable = OneDAL.makeHomogenTable(result.pcNumericTable)
          getPrincipleComponentsFromOneAPI(pcNumericTable, k, computeDevice)
        } else {
          val pcNumericTable = OneDAL.makeNumericTable(result.pcNumericTable)
          getPrincipleComponentsFromDAL(pcNumericTable, k)
        }

        val explainedVariance = if (useDevice == "GPU") {
          val explainedVarianceNumericTable = OneDAL.makeHomogenTable(
            result.explainedVarianceNumericTable)
          getExplainedVarianceFromOneAPI(
            explainedVarianceNumericTable, k, computeDevice)
        } else {
          val explainedVarianceNumericTable = OneDAL.makeNumericTable(
            result.explainedVarianceNumericTable)
          getExplainedVarianceFromDAL(explainedVarianceNumericTable, k)
        }

        Iterator((principleComponents, explainedVariance))
      } else {
        Iterator.empty
      }
      if (useDevice == "CPU") {
        OneCCL.cleanup()
      }
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

  private[mllib] def getPrincipleComponentsFromOneAPI(table: HomogenTable,
                                                      k: Int,
                                                      device: Common.ComputeDevice): DenseMatrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt
    require(k <= numRows, "k should be less or equal to row number")

    val arrayDouble = getDoubleBufferDataFromOneAPI(table, numRows, device)

    // Column-major, transpose of top K rows of NumericTable
    new DenseMatrix(numCols, k, arrayDouble.slice(0, numCols * k), false)
  }

  private[mllib] def getExplainedVarianceFromOneAPI(table_1xn: HomogenTable, k: Int,
                                          device: Common.ComputeDevice): DenseVector = {
    val arrayDouble = getDoubleBufferDataFromOneAPI(table_1xn, 1, device)
    val sum = arrayDouble.sum
    val topK = Arrays.copyOfRange(arrayDouble, 0, k)
    for (i <- 0 until k)
      topK(i) = topK(i) / sum
    new DenseVector(topK)
  }

  // table.asInstanceOf[HomogenNumericTable].getDoubleArray() would error on GPU,
  // so use table.getBlockOfRows instead of it.
  private[mllib] def getDoubleBufferDataFromOneAPI(table: HomogenTable,
                                         numRows: Int,
                                         device: Common.ComputeDevice): Array[Double] = {

    // returned DoubleBuffer is ByteByffer, need to copy as double array
    val accessor = new RowAccessor(table.getcObejct(), device)
    val arrayDouble: Array[Double] = accessor.pullDouble(0, numRows)

    arrayDouble
  }

  private def getPrincipleComponentsFromDAL(table: NumericTable, k: Int): DenseMatrix = {
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt
    require(k <= numRows, "k should be less or equal to row number")

    val arrayDouble = getDoubleBufferDataFromDAL(table, numRows, numCols)

    // Column-major, transpose of top K rows of NumericTable
    new DenseMatrix(numCols, k, arrayDouble.slice(0, numCols * k), false)
  }

  private def getExplainedVarianceFromDAL(table_1xn: NumericTable, k: Int): DenseVector = {
    val dataNumCols = table_1xn.getNumberOfColumns.toInt
    val arrayDouble = getDoubleBufferDataFromDAL(table_1xn, 1, dataNumCols)
    val sum = arrayDouble.sum
    val topK = Arrays.copyOfRange(arrayDouble, 0, k)
    for (i <- 0 until k)
      topK(i) = topK(i) / sum
    new DenseVector(topK)
  }

  // table.asInstanceOf[HomogenNumericTable].getDoubleArray() would error on GPU,
  // so use table.getBlockOfRows instead of it.
  private def getDoubleBufferDataFromDAL(table: NumericTable,
                                         numRows: Int,
                                         numCols: Int): Array[Double] = {
    var dataDouble: DoubleBuffer = null

    // returned DoubleBuffer is ByteByffer, need to copy as double array
    dataDouble = table.getBlockOfRows(0, numRows, dataDouble)
    val arrayDouble: Array[Double] = new Array[Double](numRows * numCols)
    dataDouble.get(arrayDouble)

    arrayDouble
  }


  // Single entry to call Correlation PCA DAL backend with parameter K
  @native private[mllib] def cPCATrainDAL(data: Long,
                                   executorNum: Int,
                                   executorCores: Int,
                                   computeDeviceOrdinal: Int,
                                   gpuIndices: Array[Int],
                                   result: PCAResult): Long
}
