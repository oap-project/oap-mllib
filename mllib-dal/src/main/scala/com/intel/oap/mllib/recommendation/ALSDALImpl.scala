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

package com.intel.oap.mllib.recommendation

import com.intel.daal.data_management.data.CSRNumericTable
import com.intel.daal.services.DaalContext
import com.intel.oap.mllib.Utils.getOneCCLIPPort
import com.intel.oap.mllib.{OneCCL, OneDAL, Utils}
import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.rdd.RDD

import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class ALSDataPartitioner(blocks: Int, itemsInBlock: Long)
  extends Partitioner {
  def numPartitions: Int = blocks

  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[Long]
    // itemsInBlock = numItems / partitions
    // remaining records will belog to the last partition
    // 21 => 5, 5, 5, 6
    // 46 => 11, 11, 11, 13
    math.min((k / itemsInBlock).toInt, blocks - 1)
  }
}

class ALSDALImpl[@specialized(Int, Long) ID: ClassTag]( data: RDD[Rating[ID]],
                                                        nFactors: Int,
                                                        maxIter: Int,
                                                        regParam: Double,
                                                        alpha: Double,
                                                        seed: Long
                                                      ) extends Serializable with Logging {

  def train(): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
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

    val nBlocks = executorNum

    logInfo(s"ALSDAL fit using $executorNum Executors " +
      s"for $nVectors vectors and $nFeatures features")

    val numericTables = data.repartition(executorNum)
      .setName("Repartitioned for conversion").cache()

    val kvsIPPort = getOneCCLIPPort(numericTables)

    val results = numericTables
      // Transpose the dataset
      .map { p =>
        Rating(p.item, p.user, p.rating)
      }
      .mapPartitionsWithIndex { (rank, iter) =>
        OneCCL.init(executorNum, rank, kvsIPPort)
        val rankId = OneCCL.rankID()

        println("rankId", rankId, "nVectors", nVectors, "nFeatures", nFeatures)

        val buffer = ratingsToByteBuffer(iter.toArray)
        val bufferInfo = new ALSPartitionInfo
        val shuffledBuffer = cShuffleData(buffer, nFeatures.toInt, nBlocks, bufferInfo)
        println(s"ratingsNum : ${bufferInfo.ratingsNum}; csrRowNum : ${bufferInfo.csrRowNum}; ")

        val table = OneDAL.bufferToCSRNumericTable(shuffledBuffer, bufferInfo,
          nVectors.toInt, nFeatures.toInt, nBlocks, rankId)

        val result = new ALSResult()
        cDALImplictALS(
          table.getCNumericTable, nUsers = nVectors,
          nFactors, maxIter, regParam, alpha,
          executorNum,
          executorCores,
          rankId,
          result
        )
        Iterator(result)
      }.cache()

    val usersFactorsRDD = results
      .mapPartitionsWithIndex { (index: Int, partiton: Iterator[ALSResult]) =>
        val ret = partiton.flatMap { p =>
          val userOffset = p.cUserOffset.toInt
          val usersFactorsNumTab = OneDAL.makeNumericTable(p.cUsersFactorsNumTab)
          val nRows = usersFactorsNumTab.getNumberOfRows.toInt
          val nCols = usersFactorsNumTab.getNumberOfColumns.toInt
          var buffer = FloatBuffer.allocate(nCols * nRows)
          // should use returned buffer
          buffer = usersFactorsNumTab.getBlockOfRows(0, nRows, buffer)
          (0 until nRows).map { index =>
            val array = Array.fill(nCols) {
              0.0f
            }
            buffer.get(array, 0, nCols)
            ((index + userOffset).asInstanceOf[ID], array)
          }.toIterator
        }
        ret
      }.setName("userFactors").cache()

    val itemsFactorsRDD = results
      .mapPartitionsWithIndex { (index: Int, partiton: Iterator[ALSResult]) =>
        val ret = partiton.flatMap { p =>
          val itemOffset = p.cItemOffset.toInt
          val itemsFactorsNumTab = OneDAL.makeNumericTable(p.cItemsFactorsNumTab)
          val nRows = itemsFactorsNumTab.getNumberOfRows.toInt
          val nCols = itemsFactorsNumTab.getNumberOfColumns.toInt
          var buffer = FloatBuffer.allocate(nCols * nRows)
          // should use returned buffer
          buffer = itemsFactorsNumTab.getBlockOfRows(0, nRows, buffer)
          (0 until nRows).map { index =>
            val array = Array.fill(nCols) {
              0.0f
            }
            buffer.get(array, 0, nCols)
            ((index + itemOffset).asInstanceOf[ID], array)
          }.toIterator
        }
        ret
      }.setName("itemFactors").cache()

    usersFactorsRDD.count()
    itemsFactorsRDD.count()

    (usersFactorsRDD, itemsFactorsRDD)
  }

  def ratingsToByteBuffer(ratings: Array[Rating[ID]]): ByteBuffer = {
    val buffer = ByteBuffer.allocateDirect(ratings.length * (8 + 8 + 4))
    // Use little endian
    buffer.order(ByteOrder.LITTLE_ENDIAN)
    ratings.foreach { rating =>
      buffer.putLong(rating.user.toString.toLong)
      buffer.putLong(rating.item.toString.toLong)
      buffer.putFloat(rating.rating)
    }
    buffer
  }

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
      ret += (partitionId -> (ratingsNum, csrRowNum, rowOffset))
      rowOffset = rowOffset + csrRowNum
    }

    ret
  }

  // Single entry to call Implict ALS DAL backend
  @native private def cDALImplictALS(data: Long,
                                     nUsers: Long,
                                     nFactors: Int,
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
