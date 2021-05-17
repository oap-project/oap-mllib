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

package org.apache.spark.ml.util

import com.intel.daal.data_management.data.{CSRNumericTable, HomogenNumericTable, NumericTable, RowMergedNumericTable, Matrix => DALMatrix}
import com.intel.daal.services.DaalContext
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{DenseVector, Matrices, Matrix, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}
import org.apache.spark.storage.StorageLevel

import java.lang
import java.util.logging.{Level, Logger}
import scala.collection.mutable.ArrayBuffer

object OneDAL {

  private val logger = Logger.getLogger("util.OneDAL")
  private val logLevel = Level.INFO

  def numericTableToMatrix(table: NumericTable): Matrix = {
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt
    val matrix = Matrices.zeros(numRows, numCols)
    for (row <- 0 until numRows) {
      for (col <- 0 until numCols) {
        matrix.update(row, col, table.getDoubleValue(col, row))
      }
    }
    matrix
  }

  def numericTableNx1ToVector(table: NumericTable): Vector = {
    val numRows = table.getNumberOfRows.toInt
    val internArray = new Array[Double](numRows.toInt)
    for (row <- 0 until numRows) {
      internArray(row) = table.getDoubleValue(0, row)
    }
    Vectors.dense(internArray)
  }

  // Convert DAL numeric table to array of vectors
  def numericTableToVectors(table: NumericTable): Array[Vector] = {
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt

    val resArray = new Array[Vector](numRows.toInt)

    for (row <- 0 until numRows) {
      val internArray = new Array[Double](numCols)
      for (col <- 0 until numCols) {
        internArray(col) = table.getDoubleValue(col, row)
      }
      resArray(row) = Vectors.dense(internArray)
    }

    resArray
  }



  def makeNumericTable(cData: Long): NumericTable = {

    val context = new DaalContext()
    val table = new HomogenNumericTable(context, cData)

    table
  }

  def makeNumericTable(arrayVectors: Array[OldVector]): NumericTable = {

    val numCols = arrayVectors.head.size
    val numRows: Int = arrayVectors.size

    val context = new DaalContext()
    val matrix = new DALMatrix(context, classOf[java.lang.Double],
      numCols.toLong, numRows.toLong, NumericTable.AllocationFlag.DoAllocate)

    arrayVectors.zipWithIndex.foreach {
      case (v, rowIndex) =>
        for (colIndex <- 0 until numCols) {
          cSetDouble(matrix.getCNumericTable, rowIndex, colIndex, v(colIndex))
        }
    }

    matrix
  }

  def releaseNumericTables(sparkContext: SparkContext): Unit = {
    sparkContext.getPersistentRDDs
      .filter(r => r._2.name == "numericTables")
      .foreach { rdd =>
        val numericTables = rdd._2.asInstanceOf[RDD[Long]]
        numericTables.foreach { address =>
          OneDAL.cFreeDataMemory(address)
        }
      }
  }

  private def doubleArrayToNumericTable(points: Array[Double]): NumericTable = {
    // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
    val context = new DaalContext()
    val matrixLabel = new DALMatrix(context, classOf[lang.Double],
      1, points.length, NumericTable.AllocationFlag.DoAllocate)

    // oneDAL libs should be loaded by now, loading other native libs
    logger.info("Loading native libraries")
    LibLoader.loadLibraries()

    points.zipWithIndex.foreach { case (point: Double, index: Int) =>
      cSetDouble(matrixLabel.getCNumericTable, index, 0, point)
    }

    matrixLabel
  }

  def rddDoubleToNumericTables(doubles: RDD[Double], executorNum: Int): RDD[Long] = {
    require(executorNum > 0)
    val doublesTables = doubles.repartition(executorNum).mapPartitions { it: Iterator[Double] =>
      val data = it.toArray
      // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
      val context = new DaalContext()
      val matrix = new DALMatrix(context, classOf[java.lang.Double],
        1, data.length, NumericTable.AllocationFlag.DoAllocate)
      // oneDAL libs should be loaded by now, loading other native libs
      logger.info("Loading native libraries")
      LibLoader.loadLibraries()

      data.zipWithIndex.foreach { case (value: Double, index: Int) =>
        cSetDouble(matrix.getCNumericTable, index, 0, value)
      }
      Iterator(matrix.getCNumericTable)
    }
    doublesTables.count()

    doublesTables
  }

  def rddLabeledPointToMergedTables(labeledPoints: RDD[(Vector, Double)],
                                    executorNum: Int): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    val tables = labeledPoints.repartition(executorNum).mapPartitions {
      it: Iterator[(Vector, Double)] =>
        val points: Array[(Vector, Double)] = it.toArray

        val features = points.map(_._1)
        val labels = points.map(_._2)

        val numColumns = features(0).size

        val featuresTable = if (features(0).isInstanceOf[DenseVector]) {
          vectorsToDenseNumericTable(features.toIterator, features.length, numColumns)
        } else {
          vectorsToSparseNumericTable(features, numColumns)
        }

        val labelsTable = doubleArrayToNumericTable(labels)

        Iterator((featuresTable.getCNumericTable, labelsTable.getCNumericTable))
    }
    tables.count()

    tables
  }



  def rddVectorToMergedTables(vectors: RDD[Vector], executorNum: Int): RDD[Long] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (vectors.getNumPartitions < executorNum) {
      vectors.repartition(executorNum).setName("Repartitioned for conversion").cache()
    } else {
      vectors
    }

    // Get dimensions for each partition
    val partitionDims = Utils.getPartitionDims(dataForConversion)

    // Filter out empty partitions
    val nonEmptyPartitions = dataForConversion.mapPartitionsWithIndex {
      (index: Int, it: Iterator[Vector]) => Iterator(Tuple3(partitionDims(index)._1, index, it))
    }.filter { entry => {
      entry._1 > 0
    }
    }

    // Convert to RDD[HomogenNumericTable]
    val numericTables = nonEmptyPartitions.map { entry =>
      val numRows = entry._1
      val index = entry._2
      val it = entry._3
      val numCols = partitionDims(index)._2

      logger.info(s"Partition index: $index, numCols: $numCols, numRows: $numRows")

      val table = vectorsToDenseNumericTable(it, numRows, numCols)
      table.getCNumericTable
    }.setName("numericTables").cache()

    numericTables.count()

    // Unpersist instances RDD
    if (vectors.getStorageLevel != StorageLevel.NONE) {
      vectors.unpersist()
    }

    // Coalesce partitions belonging to the same executor
    val coalescedRdd = numericTables.coalesce(executorNum,
      partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))

    val coalescedTables = coalescedRdd.mapPartitions { iter =>
      val context = new DaalContext()
      val mergedData = new RowMergedNumericTable(context)

      iter.foreach { address =>
        OneDAL.cAddNumericTable(mergedData.getCNumericTable, address)
      }
      Iterator(mergedData.getCNumericTable)
    }.cache()

    coalescedTables
  }

  private def vectorsToDenseNumericTable(it: Iterator[Vector], numRows: Int, numCols: Int): NumericTable = {
    // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
    val context = new DaalContext()
    val matrix = new DALMatrix(context, classOf[lang.Double],
      numCols.toLong, numRows.toLong, NumericTable.AllocationFlag.DoAllocate)

    // oneDAL libs should be loaded by now, loading other native libs
    logger.info("Loading native libraries")
    LibLoader.loadLibraries()

    var dalRow = 0

    it.foreach { curVector =>
      val rowArray = curVector.toArray
      OneDAL.cSetDoubleBatch(matrix.getCNumericTable, dalRow, rowArray, 1, numCols)
      dalRow += 1
    }
    matrix
  }
  private def vectorsToSparseNumericTable(vectors: Array[Vector], nFeatures: Long): CSRNumericTable = {
    require(vectors(0).isInstanceOf[SparseVector], "vectors should be sparse")

    val ratingsNum = vectors.map(_.numActives).sum
    val csrRowNum = vectors.length
    val values = Array.fill(ratingsNum) {
      0.0
    }
    val columnIndices = Array.fill(ratingsNum) {
      0L
    }
    val rowOffsets = ArrayBuffer[Long](1L)

    var indexValues = 0
    var curRow = 0L

    // Converted to one CSRNumericTable
    for (row <- 0 until vectors.length) {
      val rowVector = vectors(row)
      rowVector.foreachActive { (column, value) =>
        values(indexValues) = value
        // one-based indexValues
        columnIndices(indexValues) = column + 1

        if (row > curRow) {
          curRow = row
          // one-based indexValues
          rowOffsets += indexValues + 1
        }

        indexValues = indexValues + 1
      }
    }
    // one-based row indexValues
    rowOffsets += indexValues + 1

    val contextLocal = new DaalContext()
    val cTable = OneDAL.cNewCSRNumericTableDouble(values, columnIndices, rowOffsets.toArray,
      nFeatures, csrRowNum)
    val table = new CSRNumericTable(contextLocal, cTable)

    table
  }

  @native def cAddNumericTable(cObject: Long, numericTableAddr: Long)

  @native def cSetDouble(numTableAddr: Long, row: Int, column: Int, value: Double)

  @native def cSetDoubleBatch(numTableAddr: Long, curRows: Int, batch: Array[Double],
                              numRows: Int, numCols: Int)

  @native def cFreeDataMemory(numTableAddr: Long)

  @native def cCheckPlatformCompatibility(): Boolean

  @native def cNewCSRNumericTableFloat(data: Array[Float],
                                       colIndices: Array[Long], rowOffsets: Array[Long],
                                       nFeatures: Long, nVectors: Long): Long
  @native def cNewCSRNumericTableDouble(data: Array[Double],
                                       colIndices: Array[Long], rowOffsets: Array[Long],
                                       nFeatures: Long, nVectors: Long): Long
}
