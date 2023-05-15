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

import org.apache.spark.scheduler.{ExecutorCacheTaskLocation, TaskLocation}
import org.apache.spark.{Partition, SparkException}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class ExecutorInProcessCoalescePartitionerV2
  extends PartitionCoalescer with Serializable {

  def coalesce(maxPartitions: Int, prev: RDD[_]): Array[PartitionGroup] = {
    val map = new mutable.HashMap[String, mutable.HashSet[Partition]]()
    val groupArr = ArrayBuffer[PartitionGroup]()
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
      val pg = new PartitionGroup(Some(x._1))
      if (x._2.toList.size > 1) {
        println(s"ExecutorInProcessCoalescePartitioner x._2.toList(0) ${x._2.toList(0)}")
        println(s"ExecutorInProcessCoalescePartitioner x._2.toList(0) " +
          s"${x._2.toList(0).getClass.getName}")
        pg.partitions += x._2.toList(0)
        println(s"ExecutorInProcessCoalescePartitioner partitions ${pg.partitions.toList}")

      }
      groupArr += pg
    })
    if (groupArr.length == 0) {
      throw new SparkException(
        "ExecutorInProcessCoalescePartitioner: No partitions or no locations for partitions found.")
    }

    val sortedGroupArr = groupArr.sortWith(_.partitions(0).index < _.partitions(0).index)

    sortedGroupArr.toArray
  }
}
