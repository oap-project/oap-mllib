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

import com.intel.oneapi.dal.table.Common
import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD

object CommonJob {
  def initCCLAndSetAffinityMask(data: RDD[_],
                                      executorNum: Int,
                                      kvsIPPort: String,
                                      useDevice: String): Unit = {
        data.mapPartitionsWithIndex { (rank, table) =>
          val gpuIndices = if (useDevice == "GPU") {
              val resources = TaskContext.get().resources()
              resources("gpu").addresses.map(_.toInt)
          } else {
              Array.empty[Int]
          }
          OneCCL.init(executorNum, rank, kvsIPPort,
            Common.ComputeDevice.getDeviceByName(useDevice).ordinal())
          Iterator.empty
        }.count()
    }
}
