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
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.{Partition, SparkContext, SparkException, TaskContext}
import org.apache.spark.scheduler.{ExecutorCacheTaskLocation, TaskLocation}
import org.apache.spark.sql.SparkSession

import scala.collection.immutable.Map
import scala.collection.{breakOut, mutable}
import scala.collection.mutable.ArrayBuffer

object SparkUtils {
  def getMapping(prev: RDD[_]): mutable.HashMap[Int, String] = {
    println(s"getMapping")
    val map = new mutable.HashMap[String, mutable.HashSet[Partition]]()
    val partitionMapping = mutable.HashMap[Int, String]()
    prev.partitions.foreach(p => {
      val loc = prev.context.getPreferredLocs(prev, p.index)
      loc.foreach {
        case location: ExecutorCacheTaskLocation =>
          val execLoc = "executor_" + location.host + "_" + location.executorId
          val partValue = map.getOrElse(execLoc, new mutable.HashSet[Partition]())
          partValue.add(p)
          map.put(execLoc, partValue)
        case _: TaskLocation =>
          throw new SparkException("ExecutorInProcessCoalescePartitioner: Invalid task location!")
      }
    })
    map.foreach(x => {
      val list = x._2.toList.sortWith(_.index < _.index)
      var index = 0
      list.foreach(partitionId => {
        // executor_host_executorId_rankId
        partitionMapping.put(partitionId.index, x._1 + "_" + index)
        index += 1
      })
    })
    if (partitionMapping.size == 0) {
      throw new SparkException(
        "ExecutorInProcessCoalescePartitioner: No partitions or no locations for partitions found.")
    }
    partitionMapping
  }

  def computeEachExecutorDataSize(rowcountMapping: Map[Int, (Int, Int)],
                                  mapping: mutable.HashMap[Int, String])
  : mutable.HashMap[String, Int] = {
    println(s"computeEachExecutorDataSize")
    val executorDataSizeMapping = mutable.HashMap[String, Int]()
    val mappedMap = rowcountMapping.map { case (key1, value1) =>
      val value2 = mapping.getOrElse(key1, "default")
      key1 -> (value1, value2)
    }.foreach { case (key, (value1, value2)) =>
      println(s"Key: $key, Value1: $value1, Value2: $value2")
      val executorID = value2.substring(0, value2.lastIndexOf("_"))
      var rowNum: Int = executorDataSizeMapping.get(executorID).getOrElse(0)
      rowNum += value1._1
      executorDataSizeMapping.put(executorID, rowNum)
    }
    executorDataSizeMapping
  }
}
