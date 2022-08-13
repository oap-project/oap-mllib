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

package com.intel.oap.mllib

import com.intel.daal.data_management.data.{CSRNumericTable, HomogenNumericTable, NumericTable, RowMergedNumericTable, Matrix => DALMatrix}
import com.intel.daal.services.DaalContext
import org.apache.spark.SparkContext
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Matrix => OldMatrix, Vector => OldVector}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import java.lang
import java.nio.DoubleBuffer
import java.util.logging.{Level, Logger}

import com.intel.oneapi.dal.table.Common.ComputeDevice
import com.intel.oneapi.dal.table.{ColumnAccessor, Common, HomogenTable, RowAccessor}

import scala.collection.mutable.ArrayBuffer

import com.intel.daal.data_management.data.{
  CSRNumericTable,
  HomogenNumericTable,
  Matrix => DALMatrix,
  NumericTable,
  RowMergedNumericTable
}
import com.intel.daal.services.DaalContext
import com.intel.oneapi.dal.table.{ColumnAccessor, Common, HomogenTable, RowAccessor}

import org.apache.spark.ml.linalg.{
  DenseMatrix,
  DenseVector,
  Matrix,
  SparseVector,
  Vector,
  Vectors
}
import org.apache.spark.mllib.linalg.{
  DenseMatrix => OldDenseMatrix,
  Matrix => OldMatrix,
  Vector => OldVector
}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object OneDAL {

  LibLoader.loadLibraries()

  private val logger = Logger.getLogger("util.OneDAL")
  private val logLevel = Level.INFO

  def numericTableToMatrix(table: NumericTable): Matrix = {
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt

    var dataDouble: DoubleBuffer = null
    // returned DoubleBuffer is ByteByffer, need to copy as double array
    dataDouble = table.getBlockOfRows(0, numRows, dataDouble)
    val arrayDouble = new Array[Double](numRows * numCols)
    dataDouble.get(arrayDouble)

    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val matrix = new DenseMatrix(numRows, numCols, arrayDouble, isTransposed = true)

    table.releaseBlockOfRows(0, numRows, dataDouble)

    matrix
  }

  def homogenTableToMatrix(table: HomogenTable): Matrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt

    val arrayDouble = table.getDoubleData()
    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val matrix = new DenseMatrix(numRows, numCols, arrayDouble, isTransposed = true)

    matrix
  }

  def numericTableToOldMatrix(table: NumericTable): OldMatrix = {
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt

    var dataDouble: DoubleBuffer = null
    // returned DoubleBuffer is ByteByffer, need to copy as double array
    dataDouble = table.getBlockOfRows(0, numRows, dataDouble)
    val arrayDouble = new Array[Double](numRows * numCols)
    dataDouble.get(arrayDouble)

    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val OldMatrix = new OldDenseMatrix(numRows, numCols, arrayDouble, isTransposed = false)

    table.releaseBlockOfRows(0, numRows, dataDouble)

    OldMatrix
  }

  def homogenTableToOldMatrix(table: HomogenTable): OldMatrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt

    val arrayDouble = table.getDoubleData()
    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val OldMatrix = new OldDenseMatrix(numRows, numCols, arrayDouble, isTransposed = true)

    OldMatrix
  }

  def isDenseDataset(ds: Dataset[_], featuresCol: String): Boolean = {
    val row = ds.select(featuresCol).head()

    row.get(0).isInstanceOf[DenseVector]
  }

  def numericTableNx1ToVector(table: NumericTable): Vector = {
    val numRows = table.getNumberOfRows.toInt

    var dataDouble: DoubleBuffer = null
    // returned DoubleBuffer is ByteByffer, need to copy as double array
    dataDouble = table.getBlockOfColumnValues(0, 0, numRows, dataDouble)
    val arrayDouble = new Array[Double](numRows.toInt)
    dataDouble.get(arrayDouble)

    table.releaseBlockOfColumnValues(0, 0, numRows, dataDouble)

    Vectors.dense(arrayDouble)
  }

  def homogenTableNx1ToVector(cTable: Long, device: Common.ComputeDevice ): Vector = {
    val columnAcc = new ColumnAccessor(cTable, device)
    val arrayDouble = columnAcc.pullDouble(0)
    Vectors.dense(arrayDouble)
  }

  def numericTable1xNToVector(table: NumericTable): Vector = {
    val numCols = table.getNumberOfColumns.toInt

    var dataDouble: DoubleBuffer = null
    // returned DoubleBuffer is ByteBuffer, need to copy as double array
    dataDouble = table.getBlockOfRows(0, 1, dataDouble)
    val arrayDouble = new Array[Double](numCols.toInt)
    dataDouble.get(arrayDouble)

    table.releaseBlockOfRows(0, 1, dataDouble)

    Vectors.dense(arrayDouble)
  }

  def homogenTable1xNToVector(table: HomogenTable, device: Common.ComputeDevice): Vector = {
    val rowAcc = new RowAccessor(table.getcObejct, device)
    val arrayDouble = rowAcc.pullDouble(0, 1)
    Vectors.dense(arrayDouble)
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

  def homogenTableToVectors(table: HomogenTable, device: Common.ComputeDevice): Array[Vector] = {
    val numRows = table.getRowCount.toInt

    val rowAcc = new RowAccessor(table.getcObejct(), device)

    val resArray = new Array[Vector](numRows.toInt)

    for (row <- 0 until numRows) {
      val internArray = rowAcc.pullDouble(row, row + 1)
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

  def makeHomogenTable(cData: Long): HomogenTable = {
    val table = new HomogenTable(cData)

    table
  }

  def rddDoubleToNumericTables(doubles: RDD[Double], executorNum: Int): RDD[Long] = {
    require(executorNum > 0)

    val doublesTables = doubles.repartition(executorNum).mapPartitions { it: Iterator[Double] =>
      val data = it.toArray
      // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
      val context = new DaalContext()
      val matrix = new DALMatrix(context, classOf[java.lang.Double],
        1, data.length, NumericTable.AllocationFlag.DoAllocate)

      data.zipWithIndex.foreach { case (value: Double, index: Int) =>
        cSetDouble(matrix.getCNumericTable, index, 0, value)
      }
      Iterator(matrix.getCNumericTable)
    }
    doublesTables.count()

    doublesTables
  }

  def rddLabeledPointToSparseTables(labeledPoints: Dataset[_],
                                    labelCol: String,
                                    featuresCol: String,
                                    executorNum: Int): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    val spark = SparkSession.active
    import spark.implicits._

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (labeledPoints.rdd.getNumPartitions < executorNum) {
      labeledPoints.repartition(executorNum).cache()
    } else {
      labeledPoints
    }

    dataForConversion.cache().count()

    val labeledPointsRDD = dataForConversion.select(labelCol, featuresCol).toDF().map {
      case Row(label: Double, features: Vector) => (features, label)
    }.rdd

    val tables = labeledPointsRDD
      .coalesce(executorNum, partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))
      .mapPartitions { it: Iterator[(Vector, Double)] =>
        val points: Array[(Vector, Double)] = it.toArray

        val features = points.map(_._1)
        val labels = points.map(_._2)

        if (features.size == 0) {
          Iterator()
        } else {
          val numColumns = features(0).size
          val featuresTable = vectorsToSparseNumericTable(features, numColumns)
          val labelsTable = doubleArrayToNumericTable(labels)

          Iterator((featuresTable.getCNumericTable, labelsTable.getCNumericTable))
        }
      }.cache()

    tables.count()

    tables
  }

  def makeHomogenTable(arrayVectors: Array[Vector],
                       device: Common.ComputeDevice): HomogenTable = {
    val numCols = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( vector: Vector <- arrayVectors) {
      for (i <- 0 until vector.toArray.length ) {
        arrayDouble(index) = vector(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    val table = new HomogenTable(numRows.toLong, numCols.toLong, arrayDouble,
      device)

    table
  }

  def makeHomogenTable(arrayVectors: Array[OldVector],
                       device: Common.ComputeDevice): HomogenTable = {
    val numCols = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( vector: OldVector <- arrayVectors) {
      for (i <- 0 until vector.toArray.length ) {
        arrayDouble(index) = vector(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    val table = new HomogenTable(numRows.toLong, numCols.toLong, arrayDouble,
      device)
    table
  }

  def rddLabeledPointToSparseTables_shuffle(labeledPoints: Dataset[_],
                                            labelCol: String,
                                            featuresCol: String,
                                            executorNum: Int): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    val spark = SparkSession.active

    val labeledPointsRDD = labeledPoints.select(labelCol, featuresCol).rdd.map {
      case Row(label: Double, features: Vector) => (features, label)
    }

    // Repartition to executorNum
    val dataForConversion = labeledPointsRDD.repartition(executorNum)
      .setName("Repartitioned for conversion")

    val tables = dataForConversion.mapPartitions { it: Iterator[(Vector, Double)] =>
      val points: Array[(Vector, Double)] = it.toArray

      val features = points.map(_._1)
      val labels = points.map(_._2)

      if (features.size == 0) {
        Iterator()
      } else {
        val numColumns = features(0).size
        val featuresTable = vectorsToSparseNumericTable(features, numColumns)
        val labelsTable = doubleArrayToNumericTable(labels)

        Iterator((featuresTable.getCNumericTable, labelsTable.getCNumericTable))
      }
    }.cache()

    tables.count()

    tables
  }

  def rddLabeledPointToMergedHomogenTables(labeledPoints: Dataset[_],
                                           labelCol: String,
                                           featuresCol: String,
                                           executorNum: Int,
                                           device: Common.ComputeDevice): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")
    printf(s"Processing partitions with $executorNum executors \n")

    val spark = SparkSession.active
    import spark.implicits._

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (labeledPoints.rdd.getNumPartitions < executorNum) {
      labeledPoints.repartition(executorNum).cache()
    } else {
      labeledPoints
    }
    printf(s"rddLabeledPointToMergedHomogenTables \n")

    val tables = dataForConversion.select(labelCol, featuresCol)
      .toDF().mapPartitions { it: Iterator[Row] =>
      val rows = it.toArray

      val features = rows.map {
        case Row(label: Double, features: Vector) => features
      }

      val labels = rows.map {
        case Row(label: Double, features: Vector) => label
      }

      if (features.size == 0) {
        Iterator()
      } else {
        val numColumns = features(0).size

        val featuresTable: HomogenTable = vectorsToDenseHomogenTable(features.toIterator,
          features.length, numColumns, device)

        // TODO : After implementing CSRTable, will implement vectorsToSparseCSRTable function

        val labelsTable = doubleArrayToHomogenTable(labels, device)

        Iterator((featuresTable.getcObejct(), labelsTable.getcObejct()))
      }
    }.cache()

    tables.count()
    printf(s"rddLabeledPointToMergedHomogenTables 1\n")

    // Coalesce partitions belonging to the same executor
    val coalescedTables = tables.rdd.coalesce(executorNum,
      partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))

    val mergedTables = coalescedTables.mapPartitions { iter =>
      val mergedFeatures = new HomogenTable(device)
      val mergedLabels = new HomogenTable(device)

      iter.foreach { case (featureAddr, labelAddr) =>
        mergedFeatures.addHomogenTable(featureAddr)
        mergedLabels.addHomogenTable(labelAddr)
      }
      Iterator((mergedFeatures.getcObejct(), mergedLabels.getcObejct()))
    }.cache()

    mergedTables.count()

    mergedTables
  }

  private[mllib] def doubleArrayToHomogenTable(
      points: Array[Double],
      device: Common.ComputeDevice): HomogenTable = {
    val table = new HomogenTable(1, points.length, points, device)
    table
  }

  private def vectorsToDenseHomogenTable(
      it: Iterator[Vector],
      numRows: Int,
      numCols: Int,
      device: Common.ComputeDevice): HomogenTable = {
    printf(s"vectorsToDenseHomogenTable numRows: $numRows numCols: $numCols \n")
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    it.foreach { curVector =>
      val rowArray = curVector.toArray
      for (i <- 0 until rowArray.length) {
        arrayDouble(index) = rowArray(i)
        assert(
          index < (numCols * numRows),
          "the size of arrayDouble should be less than or equal to numCols * numRows ")
        index = index + 1
      }
    }
    val table = new HomogenTable(numRows.toLong, numCols.toLong, arrayDouble, device)
    table
  }

  def rddLabeledPointToMergedTables(labeledPoints: Dataset[_],
                                      labelCol: String,
                                      featuresCol: String,
                                      executorNum: Int): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    val spark = SparkSession.active
    import spark.implicits._

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (labeledPoints.rdd.getNumPartitions < executorNum) {
      labeledPoints.repartition(executorNum).cache()
    } else {
      labeledPoints
    }

    val tables = dataForConversion.select(labelCol, featuresCol)
      .toDF().mapPartitions { it: Iterator[Row] =>
      val rows = it.toArray

      val features = rows.map {
        case Row(label: Double, features: Vector) => features
      }

      val labels = rows.map {
        case Row(label: Double, features: Vector) => label
      }

      if (features.size == 0) {
        Iterator()
      } else {
        val numColumns = features(0).size

        val featuresTable = if (features(0).isInstanceOf[DenseVector]) {
          vectorsToDenseNumericTable(features.toIterator, features.length, numColumns)
        } else {
          vectorsToSparseNumericTable(features, numColumns)
        }

        val labelsTable = doubleArrayToNumericTable(labels)

        Iterator((featuresTable.getCNumericTable, labelsTable.getCNumericTable))
      }
    }.cache()

    tables.count()

    // Coalesce partitions belonging to the same executor
    val coalescedTables = tables.rdd.coalesce(executorNum,
      partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))

    val mergedTables = coalescedTables.mapPartitions { iter =>
      val context = new DaalContext()
      val mergedFeatures = new RowMergedNumericTable(context)
      val mergedLabels = new RowMergedNumericTable(context)

      iter.foreach { case (featureAddr, labelAddr) =>
        OneDAL.cAddNumericTable(mergedFeatures.getCNumericTable, featureAddr)
        OneDAL.cAddNumericTable(mergedLabels.getCNumericTable, labelAddr)
      }

      //      Service.printNumericTable("mergedFeatures", mergedFeatures, 10, 20)
      //      Service.printNumericTable("mergedLabels", mergedLabels, 10, 20)

      Iterator((mergedFeatures.getCNumericTable, mergedLabels.getCNumericTable))
    }.cache()

    mergedTables.count()

    mergedTables
  }

  private[mllib] def doubleArrayToNumericTable(points: Array[Double]): NumericTable = {
    // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
    val context = new DaalContext()
    val matrixLabel = new DALMatrix(
      context,
      classOf[lang.Double],
      1,
      points.length,
      NumericTable.AllocationFlag.DoAllocate)

    points.zipWithIndex.foreach {
      case (point: Double, index: Int) =>
        cSetDouble(matrixLabel.getCNumericTable, index, 0, point)
    }

    matrixLabel
  }

  def vectorsToSparseNumericTable(vectors: Array[Vector], nFeatures: Long): CSRNumericTable = {
    require(vectors(0).isInstanceOf[SparseVector], "vectors should be sparse")

    println(s"Features row x column: ${vectors.length} x ${vectors(0).size}")

    val ratingsNum = vectors.map(_.numActives).sum
    val csrRowNum = vectors.length
    val values = Array.fill(ratingsNum) {
      0.0
    }
    val columnIndices = Array.fill(ratingsNum) {
      0L
    }
    // First row index is 1
    val rowOffsets = ArrayBuffer[Long](1L)

    var indexValues = 0

    // Converted to one CSRNumericTable
    for (row <- 0 until vectors.length) {
      val rowVector = vectors(row)
      rowVector.foreachActive { (column, value) =>
        values(indexValues) = value
        // one-based indexValues
        columnIndices(indexValues) = column + 1

        indexValues = indexValues + 1
      }
      // one-based row indexValues
      rowOffsets += indexValues + 1
    }

    val contextLocal = new DaalContext()

    // check CSR encoding
    assert(
      values.length == ratingsNum,
      "the length of values should be equal to the number of non-zero elements")
    assert(
      columnIndices.length == ratingsNum,
      "the length of columnIndices should be equal to the number of non-zero elements")
    assert(
      rowOffsets.size == (csrRowNum + 1),
      "the size of rowOffsets should be equal to the number of rows + 1")

    val cTable = OneDAL.cNewCSRNumericTableDouble(
      values,
      columnIndices,
      rowOffsets.toArray,
      nFeatures,
      csrRowNum)
    val table = new CSRNumericTable(contextLocal, cTable)

    table
  }

  private def vectorsToDenseNumericTable(
                                          it: Iterator[Vector],
                                          numRows: Int,
                                          numCols: Int): NumericTable = {
    // Build DALMatrix, this will load libJavaAPI, libtbb, libtbbmalloc
    val context = new DaalContext()
    val matrix = new DALMatrix(
      context,
      classOf[lang.Double],
      numCols.toLong,
      numRows.toLong,
      NumericTable.AllocationFlag.DoAllocate)

    var dalRow = 0

    it.foreach { curVector =>
      val rowArray = curVector.toArray
      OneDAL.cSetDoubleBatch(matrix.getCNumericTable, dalRow, rowArray, 1, numCols)
      dalRow += 1
    }

    matrix
  }

  def partitionsToNumericTables(partitions: RDD[Vector], executorNum: Int): RDD[NumericTable] = {
    val dataForConversion = partitions
      .repartition(executorNum)
      .setName("Repartitioned for conversion")
      .cache()

    dataForConversion.mapPartitionsWithIndex { (index: Int, it: Iterator[Vector]) =>
      val table = makeNumericTable(it.toArray)
      Iterator(table)
    }
  }

  def makeNumericTable(arrayVectors: Array[Vector]): NumericTable = {

    val numCols = arrayVectors.head.size
    val numRows: Int = arrayVectors.size

    val context = new DaalContext()
    val matrix = new DALMatrix(context, classOf[java.lang.Double],
      numCols.toLong, numRows.toLong, NumericTable.AllocationFlag.DoAllocate)

    arrayVectors.zipWithIndex.foreach {
      case (v, rowIndex) =>
        for (colIndex <- 0 until numCols)
          cSetDouble(matrix.getCNumericTable, rowIndex, colIndex, v(colIndex))
    }

    matrix
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
    }.filter {
      _._1 > 0
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

  def rddVectorToMergedHomogenTables(vectors: RDD[Vector], executorNum: Int,
                                     device: Common.ComputeDevice): RDD[Long] = {

    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")

    // Repartition to executorNum
    val dataForConversion = vectors.repartition(executorNum)
      .setName("Repartitioned for conversion")

    // Get dimensions for each partition
    val partitionDims = Utils.getPartitionDims(dataForConversion)

    // Filter out empty partitions
    val nonEmptyPartitions = dataForConversion.mapPartitionsWithIndex {
      (index: Int, it: Iterator[Vector]) => Iterator(Tuple3(partitionDims(index)._1, index, it))
    }.filter {
      _._1 > 0
    }

    // Unpersist instances RDD
    if (vectors.getStorageLevel != StorageLevel.NONE) {
      vectors.unpersist()
    }

    // Convert to RDD[HomogenTable]
    val coalescedTables = nonEmptyPartitions.map { entry =>
      val numRows = entry._1
      val index = entry._2
      val it = entry._3
      val numCols = partitionDims(index)._2

      logger.info(s"Partition index: $index, numCols: $numCols, numRows: $numRows")

      val table = vectorsToDenseHomogenTable(it, numRows, numCols, device)

      Iterator(table.getcObejct())
    }.setName("HomogenTables").cache()

    coalescedTables
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
