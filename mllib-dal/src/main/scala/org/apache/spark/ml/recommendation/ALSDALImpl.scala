package org.apache.spark.ml.recommendation

import com.intel.daal.data_management.data.CSRNumericTable.Indexing
import org.apache.spark.rdd.{ExecutorInProcessCoalescePartitioner, RDD}

import scala.reflect.ClassTag
import com.intel.daal.data_management.data.{CSRNumericTable, HomogenNumericTable, RowMergedNumericTable, Matrix => DALMatrix}
import com.intel.daal.services.DaalContext
import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.util._

import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.mutable.ArrayBuffer
//import java.nio.DoubleBuffer
import java.nio.FloatBuffer

class ALSDataPartitioner(blocks: Int, itemsInBlock: Long)
  extends Partitioner {
  def numPartitions: Int = blocks
  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Long]
    // itemsInBlock = numItems / partitions
    // remaining records will belog to the last partition
    // 21 => 5, 5, 5, 6
    // 46 => 11, 11, 11, 13
    math.min((k / itemsInBlock).toInt, blocks-1)
  }
}

class ALSDALImpl[@specialized(Int, Long) ID: ClassTag](
  data: RDD[Rating[ID]],
  rank: Int,
  maxIter: Int,
  regParam: Double,
  alpha: Double,
  seed: Long,
) extends Serializable with Logging {

  // Rating struct size is size of Long+Long+Float
  val RATING_SIZE = 8 + 8 + 4

  // Return Map partitionId -> (ratingsNum, csrRowNum, rowOffset)
  private def getRatingsPartitionInfo(data: RDD[Rating[ID]]): Map[Int, (Int, Int, Int)] = {
    val collectd = data.mapPartitionsWithIndex { case (index: Int, it: Iterator[Rating[ID]]) =>
      var ratingsNum = 0
      var s = Set[ID]()
      it.foreach { v =>
        s += v.user
        ratingsNum += 1
      }
      Iterator((index, (ratingsNum, s.count(_ => true))))
    }.collect

    var ret = Map[Int, (Int, Int, Int)]()
    var rowOffset = 0
    collectd.foreach { v =>
      val partitionId = v._1
      val ratingsNum = v._2._1
      val csrRowNum = v._2._2
      ret += ( partitionId -> (ratingsNum, csrRowNum, rowOffset))
      rowOffset = rowOffset + csrRowNum
    }

    ret
  }

  private def ratingsToCSRNumericTables(ratings: RDD[Rating[ID]],
    nVectors: Long, nFeatures: Long, nBlocks: Long): RDD[CSRNumericTable] = {

//    val rowSortedRatings = ratings.sortBy(_.user.toString.toLong)

//    val itemsInBlock = (nFeatures + nBlocks - 1) / nBlocks
    val itemsInBlock = nFeatures / nBlocks
//    val rowSortedGrouped = rowSortedRatings.groupBy(value => value.user.toString.toLong / itemsInBlock).flatMap(_._2)
    val rowSortedGrouped = ratings
      // Transpose the dataset
      .map { p =>
        Rating(p.item, p.user, p.rating)
      }
      .groupBy(value => value.user.toString.toLong)
      .partitionBy(new ALSDataPartitioner(nBlocks.toInt, itemsInBlock))
      .flatMap(_._2).mapPartitions { p =>
        p.toArray.sortBy(_.user.toString.toLong).toIterator
      }

    println("rowSortedGrouped partition number: ",  rowSortedGrouped.getNumPartitions)

    //    rowSortedGrouped.mapPartitionsWithIndex { case (partitionId, partition) =>
//        println("partitionId", partitionId)
//        partition.foreach { p =>
//          println(p.user, p.item, p.rating) }
//        Iterator(partitionId)
//    }.collect()

    val ratingsPartitionInfo = getRatingsPartitionInfo(rowSortedGrouped)
    println("ratingsPartitionInfo:",  ratingsPartitionInfo)

    rowSortedGrouped.mapPartitionsWithIndex { case (partitionId, partition) =>
      val ratingsNum = ratingsPartitionInfo(partitionId)._1
      val csrRowNum = ratingsPartitionInfo(partitionId)._2
      val values = Array.fill(ratingsNum) { 0.0f }
      val columnIndices = Array.fill(ratingsNum) { 0L }
      val rowOffsets = ArrayBuffer[Long](1L)


      var index = 0
      var curRow = 0L
      // Each partition converted to one CSRNumericTable
      partition.foreach { p =>
        // Modify row index for each partition (start from 0)
        val row = p.user.toString.toLong - ratingsPartitionInfo(partitionId)._3
        val column = p.item.toString.toLong
        val rating = p.rating

        values(index) = rating
        // one-based index
        columnIndices(index) = column + 1

        if (row > curRow) {
          curRow = row
          // one-based index
          rowOffsets += index + 1
        }

        index = index + 1
      }
      // one-based row index
      rowOffsets += index+1

      println("PartitionId:", partitionId)
      println("csrRowNum", csrRowNum)
//      println("rowOffsets", rowOffsets.mkString(","))
//      println("columnIndices", columnIndices.mkString(","))
//      println("values", values.mkString(","))

      val contextLocal = new DaalContext()

      println("ALSDALImpl: Loading native libraries ..." )
      LibLoader.loadLibraries()

      val cTable = OneDAL.cNewCSRNumericTable(values, columnIndices, rowOffsets.toArray, nVectors, csrRowNum)
      val table = new CSRNumericTable(contextLocal, cTable)
//      table.pack()

      println("Input dimensions:", table.getNumberOfRows, table.getNumberOfColumns)

      // There is a bug https://github.com/oneapi-src/oneDAL/pull/1288,
      // printNumericTable can't print correct result for CSRNumericTable, use C++ printNumericTable
      // Service.printNumericTable("Input: ", table)

      Iterator(table)
    }.cache()
  }

//  def factorsToRDD(cUsersFactorsNumTab: Long, cItemsFactorsNumTab: Long)
//    :(RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
//    val usersFactorsNumTab = OneDAL.makeNumericTable(cUsersFactorsNumTab)
//    val itemsFactorsNumTab = OneDAL.makeNumericTable(cItemsFactorsNumTab)
//
//    Service.printNumericTable("usersFactorsNumTab", usersFactorsNumTab)
//    Service.printNumericTable("itemsFactorsNumTab", itemsFactorsNumTab)
//
//    null
//  }

  def ratingsToByteBuffer(ratings: Array[Rating[ID]]): ByteBuffer = {
//    println("ratings len", ratings.length)

    val buffer= ByteBuffer.allocateDirect(ratings.length*(8+8+4))
    // Use little endian
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    ratings.foreach { rating =>
      buffer.putLong(rating.user.toString.toLong)
      buffer.putLong(rating.item.toString.toLong)
      buffer.putFloat(rating.rating)
    }
    buffer
  }

  def run(): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
    val executorNum = Utils.sparkExecutorNum(data.sparkContext)
    val executorCores = Utils.sparkExecutorCores()

    val nFeatures = data.max()(new Ordering[Rating[ID]]() {
      override def compare(x: Rating[ID], y: Rating[ID]): Int =
        Ordering[Long].compare(x.item.toString.toLong, y.item.toString.toLong)
    }).item.toString.toLong + 1

    val nVectors = data.max()(new Ordering[Rating[ID]]() {
      override def compare(x: Rating[ID], y: Rating[ID]): Int =
        Ordering[Long].compare(x.user.toString.toLong, y.user.toString.toLong)
    }).user.toString.toLong + 1

//    val largestItems = data.sortBy(_.item.toString.toLong, ascending = false).take(1)
//    val nFeatures = largestItems(0).item.toString.toLong + 1

//    val largestUsers = data.sortBy(_.user.toString.toLong, ascending = false).take(1)
//    val nVectors = largestUsers(0).user.toString.toLong + 1

    val nBlocks = executorNum

//    val nRatings = data.count()

    logInfo(s"ALSDAL fit using $executorNum Executors for $nVectors vectors and $nFeatures features")

    val executorIPAddress = Utils.sparkFirstExecutorIP(data.sparkContext)

    val numericTables = if (data.getNumPartitions < executorNum) {
      data.repartition(executorNum).setName("Repartitioned for conversion").cache()
    } else {
      data.coalesce(executorNum).setName("Coalesced for conversion").cache()
    }

    val results = numericTables
      // Transpose the dataset
      .map { p =>
        Rating(p.item, p.user, p.rating) }
      .mapPartitions { iter =>
        val context = new DaalContext()
        println("ALSDALImpl: Loading libMLlibDAL.so" )
        LibLoader.loadLibraries()

        OneCCL.init(executorNum, executorIPAddress, OneCCL.KVS_PORT)
        val rankId = OneCCL.rankID()

        println("rankId", rankId, "nUsers", nVectors, "nItems", nFeatures)

        val buffer = ratingsToByteBuffer(iter.toArray)
        val bufferInfo = new ALSPartitionInfo
        val shuffledBuffer = cShuffleData(buffer, nFeatures.toInt, nBlocks, bufferInfo)

        val table = bufferToCSRNumericTable(shuffledBuffer, bufferInfo, nVectors.toInt, nFeatures.toInt, nBlocks, rankId)

        val result = new ALSResult()
        cDALImplictALS(
          table.getCNumericTable, nUsers = nVectors,
          rank, maxIter, regParam, alpha,
          executorNum,
          executorCores,
          rankId,
          result
        )
        Iterator(result)
    }.cache()

//    results.foreach { p =>
////      val usersFactorsNumTab = OneDAL.makeNumericTable(p.cUsersFactorsNumTab)
////      println("foreach", p.cUsersFactorsNumTab, p.cItemsFactorsNumTab)
//      println("result", p.rankId, p.cUserOffset, p.cItemOffset);
//    }

//    val usersFactorsRDD = results.mapPartitionsWithIndex { (index: Int, partiton: Iterator[ALSResult]) =>
//      partiton.foreach { p =>
//        val usersFactorsNumTab = OneDAL.makeNumericTable(p.cUsersFactorsNumTab)
//        Service.printNumericTable("usersFactorsNumTab", usersFactorsNumTab)
//      }
//      Iterator()
//    }.collect()

    val usersFactorsRDD = results.mapPartitionsWithIndex { (index: Int, partiton: Iterator[ALSResult]) =>
      val ret = partiton.flatMap { p =>
        val userOffset = p.cUserOffset.toInt
        val usersFactorsNumTab = OneDAL.makeNumericTable(p.cUsersFactorsNumTab)
        val nRows = usersFactorsNumTab.getNumberOfRows.toInt
        val nCols = usersFactorsNumTab.getNumberOfColumns.toInt
        var buffer = FloatBuffer.allocate(nCols * nRows)
        // should use returned buffer
        buffer = usersFactorsNumTab.getBlockOfRows(0, nRows, buffer)
        (0 until nRows).map { index =>
          val array = Array.fill(nCols){0.0f}
          buffer.get(array, 0, nCols)
          ((index+userOffset).asInstanceOf[ID], array)
        }.toIterator
      }
      ret
    }.setName("userFactors").cache()

    val itemsFactorsRDD = results.mapPartitionsWithIndex { (index: Int, partiton: Iterator[ALSResult]) =>
      val ret = partiton.flatMap { p =>
        val itemOffset = p.cItemOffset.toInt
        val itemsFactorsNumTab = OneDAL.makeNumericTable(p.cItemsFactorsNumTab)
        val nRows = itemsFactorsNumTab.getNumberOfRows.toInt
        val nCols = itemsFactorsNumTab.getNumberOfColumns.toInt
        var buffer = FloatBuffer.allocate(nCols * nRows)
        // should use returned buffer
        buffer = itemsFactorsNumTab.getBlockOfRows(0, nRows, buffer)
        (0 until nRows).map { index =>
          val array = Array.fill(nCols){0.0f}
          buffer.get(array, 0, nCols)
          ((index+itemOffset).asInstanceOf[ID], array)
        }.toIterator
      }
      ret
    }.setName("itemFactors").cache()

    usersFactorsRDD.count()
    itemsFactorsRDD.count()

//    usersFactorsRDD.foreach { case (id, array) =>
//        println("usersFactorsRDD", id, array.mkString(", "))
//    }
//
//    itemsFactorsRDD.foreach { case (id, array) =>
//      println("itemsFactorsRDD", id, array.mkString(", "))
//    }

    (usersFactorsRDD, itemsFactorsRDD)
  }

  private def getPartitionOffset(partitionId: Int, nRatings: Int, nBlocks: Int): Int = {
    require(partitionId >=0 && partitionId < nBlocks)
    val itemsInBlock = nRatings / nBlocks
    return partitionId * itemsInBlock
  }

  private def bufferToCSRNumericTable(buffer: ByteBuffer, info: ALSPartitionInfo,
                                      nVectors: Int, nFeatures: Int, nBlocks: Int, rankId: Int): CSRNumericTable = {
    // Use little endian
    buffer.order(ByteOrder.LITTLE_ENDIAN)

    val ratingsNum = info.ratingsNum
    val csrRowNum = info.csrRowNum
    val values = Array.fill(ratingsNum) { 0.0f }
    val columnIndices = Array.fill(ratingsNum) { 0L }
    val rowOffsets = ArrayBuffer[Long](1L)

    var index = 0
    var curRow = 0L
    // Each partition converted to one CSRNumericTable
    for (i <- 0 until ratingsNum) {
      // Modify row index for each partition (start from 0)
      val row = buffer.getLong(i*RATING_SIZE) - getPartitionOffset(rankId, nFeatures, nBlocks)
      val column = buffer.getLong(i*RATING_SIZE+8)
      val rating = buffer.getFloat(i*RATING_SIZE+16)

      values(index) = rating
      // one-based index
      columnIndices(index) = column + 1

      if (row > curRow) {
        curRow = row
        // one-based index
        rowOffsets += index + 1
      }

      index = index + 1
    }
    // one-based row index
    rowOffsets += index+1

//    println("rankId:", rankId)
//    println("csrRowNum", csrRowNum)

//    println(rowOffsets.mkString(" "))
//    println(columnIndices.mkString(" "))
//    println(values.mkString(" "))

    val contextLocal = new DaalContext()
    val cTable = OneDAL.cNewCSRNumericTable(values, columnIndices, rowOffsets.toArray, nVectors, csrRowNum)
    val table = new CSRNumericTable(contextLocal, cTable)

    println("Input dimensions:", table.getNumberOfRows, table.getNumberOfColumns)
//    Service.printNumericTable("Input NumericTable", table)

    table
  }

  // Single entry to call Implict ALS DAL backend
  @native private def cDALImplictALS(data: Long, 
                                     nUsers: Long,
                                     rank: Int,
                                     maxIter: Int,
                                     regParam: Double,
                                     alpha: Double,
                                     executor_num: Int,
                                     executor_cores: Int,
                                     rankId: Int,
                                     result: ALSResult): Long
  @native private def cShuffleData(data: ByteBuffer,
                                   nTotalKeys: Int,
                                   nBlocks: Int,
                                   info: ALSPartitionInfo): ByteBuffer
}