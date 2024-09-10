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

import org.apache.spark.internal.Logging

object OneCCL extends Logging {

  LibLoader.loadLibraries()

  var cclParam = new CCLParam()

  def init(executor_num: Int, rank: Int, ip_port: String): Unit = {

    logInfo(s"Initializing with IP_PORT: ${ip_port}")

    // cclParam is output from native code
    c_init(executor_num, rank, ip_port, cclParam)

    // executor number should equal to oneCCL world size
    assert(executor_num == cclParam.getCommSize,
      "executor number should equal to oneCCL world size")

    logInfo(s"Initialized with executorNum: $executor_num, " +
      s"commSize, ${cclParam.getCommSize}, rankId: ${cclParam.getRankId}")
  }

  // Set specified by values for each Executor
  def setAffinityMask(rankId: String): Unit = {
      setEnv("ZE_AFFINITY_MASK", rankId)
  }

  // Run on Executor
  def cleanup(): Unit = {
    c_cleanup()
  }

  def getAvailPort(localIP: String): Int = synchronized {
    c_getAvailPort(localIP)
  }

  @native def isRoot(): Boolean

  @native def rankID(): Int

  @native def setEnv(key: String, value: String, overwrite: Boolean = true): Int

  @native def c_getAvailPort(localIP: String): Int

  @native private def c_init(size: Int, rank: Int, ip_port: String, param: CCLParam): Int

  @native private def c_cleanup(): Unit
}
