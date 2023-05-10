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

import org.apache.spark.broadcast.Broadcast

import scala.collection.mutable

object ExecutorSharedArray {
  @transient private var sharedArray: Array[Double] = _

  def getSharedArray(numCols: Int,
                     bcMapping: Broadcast[mutable.HashMap[Int, String]],
                     bcRowcount: Broadcast[Map[String, Int]],
                     partitionId: Int): Array[Double] = {
    println(s"computeAndCreateArray merge table start")
    if (sharedArray == null) {
      synchronized {
        if (sharedArray == null) {
          // This block of code will be executed only once per executor
          val executorId: String = bcMapping.value(partitionId)
          val id = executorId.substring(0, executorId.lastIndexOf("_"))
          println(s"computeAndCreateArray @transient lazy val executorId: ${executorId}")
          val rowcount = bcRowcount.value(id)
          println(s"computeAndCreateArray @transient lazy val rowcount: ${rowcount}")
          println(s"computeAndCreateArray @transient lazy val array size : ${numCols * rowcount}")
          sharedArray = new Array[Double](numCols * rowcount)
        }
      }
    }
    sharedArray
  }
}
