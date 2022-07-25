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
    val coalescedTables = OneDAL.rddVectorToMergedHomogenTables(normalizedData, executorNum,
      computeDevice)
    val kvsIPPort = getOneCCLIPPort(coalescedTables)

    val results = coalescedTables.mapPartitionsWithIndex { (rank, table) =>
      val tableArr = table.next()
      OneCCL.initDpcpp()

      val result = new PCAResult()
      cPCATrainDAL(
        tableArr,
        executorNum,
        computeDevice.ordinal(),
        rank,
        kvsIPPort,
        result
      )

      val ret = if (rank == 0) {
        val pcNumericTable = OneDAL.makeHomogenTable(result.pcNumericTable)
        val explainedVarianceNumericTable = OneDAL.makeHomogenTable(
          result.explainedVarianceNumericTable)
        val principleComponents = getPrincipleComponentsFromDAL(pcNumericTable, k, computeDevice)
        val explainedVariance = getExplainedVarianceFromDAL(
          explainedVarianceNumericTable, k, computeDevice)

        Iterator((principleComponents, explainedVariance))
      } else {
        Iterator.empty
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

  private[mllib] def getPrincipleComponentsFromDAL(table: HomogenTable,
                                            k: Int,
                                            device: Common.ComputeDevice): DenseMatrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt
    require(k <= numRows, "k should be less or equal to row number")

    val arrayDouble = getDoubleBufferDataFromDAL(table, numRows, device)

    // Column-major, transpose of top K rows of NumericTable
    new DenseMatrix(numCols, k, arrayDouble.slice(0, numCols * k), false)
  }

  private[mllib] def getExplainedVarianceFromDAL(table_1xn: HomogenTable, k: Int,
                                          device: Common.ComputeDevice): DenseVector = {
    val arrayDouble = getDoubleBufferDataFromDAL(table_1xn, 1, device)
    val sum = arrayDouble.sum
    val topK = Arrays.copyOfRange(arrayDouble, 0, k)
    for (i <- 0 until k)
      topK(i) = topK(i) / sum
    new DenseVector(topK)
  }

  // table.asInstanceOf[HomogenNumericTable].getDoubleArray() would error on GPU,
  // so use table.getBlockOfRows instead of it.
  private[mllib] def getDoubleBufferDataFromDAL(table: HomogenTable,
                                         numRows: Int,
                                         device: Common.ComputeDevice): Array[Double] = {

    // returned DoubleBuffer is ByteByffer, need to copy as double array
    val accessor = new RowAccessor(table.getcObejct(), device)
    val arrayDouble: Array[Double] = accessor.pullDouble(0, numRows)

    arrayDouble
  }

  // Single entry to call Correlation PCA DAL backend with parameter K
  @native private[mllib] def cPCATrainDAL(data: Long,
                                   executorNum: Int,
                                   computeDeviceOrdinal: Int,
                                   rankId: Int,
                                   ipPort: String,
                                   result: PCAResult): Long
}
