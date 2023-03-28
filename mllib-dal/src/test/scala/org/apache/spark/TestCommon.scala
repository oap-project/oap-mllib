/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.spark

import com.intel.oneapi.dal.table.{Common, HomogenTable, RowAccessor}
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}
import org.apache.spark.mllib.linalg.{Vector => OldVector}

import scala.io.Source

object TestCommon {
  def getComputeDevice: Common.ComputeDevice = {
    val device = System.getProperty("computeDevice")
    var computeDevice: Common.ComputeDevice = Common.ComputeDevice.CPU
    if(device != null) {
      device.toUpperCase match {
        case "CPU" => computeDevice = Common.ComputeDevice.CPU
        case "GPU" => computeDevice = Common.ComputeDevice.GPU
        case _ => "Invalid Device"
      }
    }
    System.out.println("getDevice : " + computeDevice)
    computeDevice
  }
}
