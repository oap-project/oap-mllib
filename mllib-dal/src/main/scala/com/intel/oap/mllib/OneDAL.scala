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

import org.apache.spark.{Partition, SparkContext, SparkException}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseVector, Vector, Vectors}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, PartitionGroup, RDD}
import org.apache.spark.storage.StorageLevel

import java.lang
import java.nio.DoubleBuffer
import java.util.logging.{Level, Logger}
import com.intel.oneapi.dal.table.Common.ComputeDevice
import com.intel.oneapi.dal.table.{ColumnAccessor, Common, HomogenTable, RowAccessor, Table}
import com.intel.daal.data_management.data.{CSRNumericTable, HomogenNumericTable, NumericTable, RowMergedNumericTable, Matrix => DALMatrix}
import com.intel.daal.services.DaalContext
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrix, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Matrix => OldMatrix, Vector => OldVector}
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}
import org.apache.spark.scheduler.{ExecutorCacheTaskLocation, TaskLocation}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable.{ArrayBuffer, ListBuffer}
import scala.collection.mutable
import scala.concurrent._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.{Duration, DurationInt}
import scala.concurrent.{Await, Future}


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

  def homogenTableToMatrix(table: HomogenTable, device: Common.ComputeDevice): Matrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt

    val accessor = new RowAccessor(table.getcObejct(), device)
    val arrayDouble: Array[Double] = accessor.pullDouble(0, numRows)

    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val matrix = new DenseMatrix(numRows, numCols, arrayDouble, isTransposed = true)

    matrix
  }

  def homogenTableToOldMatrix(table: HomogenTable, device: Common.ComputeDevice): OldMatrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt

    val accessor = new RowAccessor(table.getcObejct(), device)
    val arrayDouble: Array[Double] = accessor.pullDouble(0, numRows)

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

  def coalesceSparseLabelPointsToSparseNumericTables(labeledPoints: Dataset[_],
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

  private[mllib] def doubleArrayToHomogenTable(
      points: Array[Double],
      device: Common.ComputeDevice): HomogenTable = {
    val table = new HomogenTable(points.length,1, points, device)
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

  def coalesceLabelPointsToNumericTables(labeledPoints: Dataset[_],
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

  def coalesceLabelPointsToHomogenTables(labeledPoints: Dataset[_],
                                    labelCol: String,
                                    featuresCol: String,
                                    executorNum: Int,
                                    device: Common.ComputeDevice): RDD[(Long, Long)] = {
    require(executorNum > 0)

    logger.info(s"Processing partitions with $executorNum executors")
    val numberCores: Int  = labeledPoints.sparkSession.sparkContext.getConf.getInt("spark.executor.cores", 1)

    val spark = SparkSession.active
    import spark.implicits._
    val labeledPointsRDD = labeledPoints.rdd

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (labeledPointsRDD.getNumPartitions < executorNum) {
      logger.info(s"Repartition to executorNum if not enough partitions")
      val rePartitions = labeledPoints.repartition(executorNum).cache()
      rePartitions.count()
      rePartitions
    } else {
      labeledPoints
    }

    // Get dimensions for each partition
    val partitionDims = Utils.getPartitionDims(dataForConversion.select(featuresCol).rdd.map{ row =>
      val vector = row.getAs[Vector](0)
      vector
    })

    // Filter out empty partitions, if there is no such rdd, coalesce will report an error
    // "No partitions or no locations for partitions found".
    // TODO: ML-312: Improve ExecutorInProcessCoalescePartitioner
    val nonEmptyPartitions = dataForConversion.select(labelCol, featuresCol).toDF().rdd.mapPartitionsWithIndex {
      (index: Int, it: Iterator[Row]) => Iterator(Tuple3(partitionDims(index)._1, index, it))
    }.filter {
      _._1 > 0
    }.flatMap {
      entry => entry._3
    }

    val coalescedRdd = nonEmptyPartitions.coalesce(executorNum,
      partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))
      .setName("coalescedRdd")

    // convert RDD to HomogenTable
    val coalescedTables = coalescedRdd.mapPartitionsWithIndex { (index: Int, it: Iterator[Row]) =>
      val list = it.toList
      val subRowCount: Int = list.size / numberCores
      val labeledPointsList: ListBuffer[Future[(Array[Double], Long)]] =
        new ListBuffer[Future[(Array[Double], Long)]]()
      val numRows = list.size
      val numCols = list(0).getAs[Vector](1).toArray.size

      val labelsArray = new Array[Double](numRows)
      val featuresAddress= OneDAL.cNewDoubleArray(numRows.toLong * numCols)
      for ( i <- 0 until numberCores) {
        val f = Future {
          val iter = list.iterator
          val slice = if (i == numberCores - 1) {
            iter.slice(subRowCount * i, numRows)
          } else {
            iter.slice(subRowCount * i, subRowCount * i + subRowCount)
          }
          slice.toArray.zipWithIndex.map { case (row, index) =>
            val length = row.getAs[Vector](1).toArray.length
            OneDAL.cCopyDoubleArrayToNative(featuresAddress, row.getAs[Vector](1).toArray, subRowCount.toLong * numCols * i + length * index)
            labelsArray(subRowCount * i +  index) = row.getAs[Double](0)
          }
          (labelsArray, featuresAddress)
        }
        labeledPointsList += f

        val result = Future.sequence(labeledPointsList)
        Await.result(result, Duration.Inf)
      }
      val labelsTable = new HomogenTable(numRows.toLong, 1, labelsArray,
        device)
      val featuresTable = new HomogenTable(numRows.toLong, numCols.toLong, featuresAddress, Common.DataType.FLOAT64,
        device)

      Iterator((featuresTable.getcObejct(), labelsTable.getcObejct()))
    }.setName("coalescedTables").cache()

   coalescedTables.count()
   coalescedTables
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

  def coalesceVectorsToHomogenTables(data: RDD[Vector], executorNum: Int,
                                device: Common.ComputeDevice): RDD[Long] = {
    logger.info(s"Processing partitions with $executorNum executors")
    val numberCores: Int  = data.sparkContext.getConf.getInt("spark.executor.cores", 1)

    // Repartition to executorNum if not enough partitions
    val dataForConversion = if (data.getNumPartitions < executorNum) {
      logger.info(s"Repartition to executorNum if not enough partitions")
      val reData = data.repartition(executorNum).setName("RepartitionedRDD")
      reData.cache().count()
      reData
    } else {
      data
    }

    // Get dimensions for each partition
    val partitionDims = Utils.getPartitionDims(dataForConversion)

    // Filter out empty partitions, if there is no such rdd, coalesce will report an error
    // "No partitions or no locations for partitions found".
    // TODO: ML-312: Improve ExecutorInProcessCoalescePartitioner
    val nonEmptyPartitions = dataForConversion.mapPartitionsWithIndex {
      (index: Int, it: Iterator[Vector]) => Iterator(Tuple3(partitionDims(index)._1, index, it))
    }.filter {
      _._1 > 0
    }.flatMap {
      entry => entry._3
    }

    val coalescedRdd = nonEmptyPartitions.coalesce(executorNum,
      partitionCoalescer = Some(new ExecutorInProcessCoalescePartitioner()))
      .setName("coalescedRdd")

    // convert RDD to HomogenTable
    val coalescedTables = coalescedRdd.mapPartitionsWithIndex { (index: Int, it: Iterator[Vector]) =>
      val list = it.toList
      val subRowCount: Int = list.size / numberCores
      val futureList: ListBuffer[Future[Long]] = new ListBuffer[Future[Long]]()
      val numRows = list.size
      val numCols = list(0).toArray.size
      val size = numRows.toLong * numCols.toLong
      val targetArrayAddress = OneDAL.cNewDoubleArray(size)
      for ( i <- 0 until numberCores) {
        val f = Future {
          val iter = list.iterator
          val slice = if (i == numberCores - 1) {
            iter.slice(subRowCount * i, numRows)
          } else {
            iter.slice(subRowCount * i, subRowCount * i + subRowCount)
          }
          slice.toArray.zipWithIndex.map { case (vector, index) =>
            val length = vector.toArray.length
            OneDAL.cCopyDoubleArrayToNative(targetArrayAddress, vector.toArray, subRowCount.toLong * numCols * i + length * index)
          }
          targetArrayAddress
        }
        futureList += f

      val result = Future.sequence(futureList)
      Await.result(result, Duration.Inf)
      }
      val table = new HomogenTable(numRows.toLong, numCols.toLong, targetArrayAddress,
        Common.DataType.FLOAT64, device)

      Iterator(table.getcObejct())
    }.setName("coalescedTables").cache()
    coalescedTables.count()
    // Unpersist instances RDD
    if (data.getStorageLevel != StorageLevel.NONE) {
      data.unpersist()
    }
    coalescedTables
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

  def coalesceVectorsToNumericTables(vectors: RDD[Vector], executorNum: Int): RDD[Long] = {
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

  @native def cNewDoubleArray(size: Long): Long

  @native def cCopyDoubleArrayToNative(arrayAddr: Long,
                                 data: Array[Double],
                                 index: Long): Unit

  @native def cSetCppLoggerConf(enable: Boolean)

}
