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
package org.apache.spark.rdd

import com.intel.oap.mllib.OneDAL.logger
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.{Partition, SparkContext, SparkException, TaskContext}
import org.apache.spark.scheduler.{ExecutorCacheTaskLocation, TaskLocation}
import org.apache.spark.sql.SparkSession

import scala.collection.{breakOut, mutable}
import scala.collection.mutable.ArrayBuffer

object SparkUtils {
  def getMapping(prev: RDD[_]) : mutable.HashMap[Int, String] = {
    println(s"getMapping")
    val map = new mutable.HashMap[String, mutable.HashSet[Partition]]()
    val partitionMapping = mutable.HashMap[Int, String]()
    prev.partitions.foreach(p => {
      val loc = prev.context.getPreferredLocs(prev, p.index)
      loc.foreach {
        case location : ExecutorCacheTaskLocation =>
          val execLoc = "executor_" + location.host + "_" + location.executorId
          val partValue = map.getOrElse(execLoc, new mutable.HashSet[Partition]())
          partValue.add(p)
          map.put(execLoc, partValue)
        case _ : TaskLocation =>
          throw new SparkException("ExecutorInProcessCoalescePartitioner: Invalid task location!")
      }
    })
    map.foreach(x => {
      val list = x._2.toList.sortWith(_.index < _.index)
      var index = 0
      list.foreach( partitionId => {
        partitionMapping.put(partitionId.index, x._1 + "_" + index)
        index += 1
      })
    })
    if (partitionMapping.size == 0) {
      throw new SparkException(
        "ExecutorInProcessCoalescePartitioner: No partitions or no locations for partitions found.")
    }
    for((k, v) <- partitionMapping) {
      println(s"key: $k, value: $v")
    }
    partitionMapping
  }

  def computeEachExecutorDataSize(prev: RDD[_], mapping: mutable.HashMap[Int, String])
  : RDD[(String, Int)] = {
    println(s"computeEachExecutorDataSize")
    // Get the partition size and create the array accordingly
    val executorSize = prev.mapPartitionsWithIndex((partitionId, iter) => {
      val rowcount = iter.length
      Iterator((partitionId, rowcount))
    }).map(iter => {
      val value : String = mapping.get(iter._1).getOrElse(null)
      val executorID = value.substring(0, value.lastIndexOf("_"))
      (executorID, iter._2)
    }).reduceByKey(_ + _)
    println(s"computeEachExecutorDataSize print rdd")
    executorSize.foreach(println)
    executorSize
  }

  def computeAndCreateArray(mapping: mutable.HashMap[Int, String],
                            executorSize: RDD[(String, Int)],
                            partitionId: Int): Array[Double] = {
    println(s"computeAndCreateArray merge table start")
    @transient lazy val array: Array[Double] = {
      val executorId: String = mapping.get(partitionId).getOrElse(null)
      println(s"computeAndCreateArray @transient lazy val executorId: ${executorId}")
      val result = executorSize.mapPartitions { iter =>
          var rowcount = 0
          while (iter.hasNext) {
            val (key, value) = iter.next()
            val id = executorId.substring(0, executorId.lastIndexOf("_"))
            if (key.equals(id)) {
              rowcount = value
            }
          }
          Iterator(rowcount)
      }.collect()

      // Make sure there is only one result from rank 0
      assert(result.length == 1)
      val rowcount: Int = result(0)
      println(s"computeAndCreateArray @transient lazy val executorId: ${executorId} ," +
        s"executor rowcount: ${rowcount}")
      val executorArray: Array[Double] = new Array[Double](rowcount)
      executorArray
    }
    array
  }
}
