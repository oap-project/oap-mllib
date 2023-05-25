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

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SPARK_VERSION, SparkConf, SparkContext}
import java.net.InetAddress

object Utils {

  Class AlgoTimeStamp {
    private val _stampName = String()
    private val _nextIntervalName = String()
    private val _timeStamp = Int
  }

  class AlgoTimeMetrics {
    private var _algoTimeStampList = List[AlgoTimeStamp]()
    private var _algoName = String()

    def print(): = {

    }
  }

  class TimeMetricsTable {
    private var _algoTimeMetricsList = Map[String, AlgoTimeMetrics]()
    def print(): = {

    }
  }

  val DefaultComputeDevice = "GPU"
  def isOAPEnabled(): Boolean = {
    val sc = SparkSession.active.sparkContext
    val isDynamicAllication = sc.getConf.getBoolean("spark.dynamicAllocation.enabled", false)
    val isOap = sc.getConf.getBoolean("spark.oap.mllib.enabled", true)
    if (isOap && isDynamicAllication) {
      throw new Exception(
        s"OAP MLlib does not support dynamic allocation, " +
          s"spark.dynamicAllocation.enabled should be set to false")
    }
    isOap
  }

  def getOneCCLIPPort(data: RDD[_]): String = {
    val executorIPAddress = Utils.sparkFirstExecutorIP(data.sparkContext)
    val kvsIP = data.sparkContext.getConf.get("spark.oap.mllib.oneccl.kvs.ip",
      executorIPAddress)
    //  TODO: right now we use a configured port, will optimize auto port detection
    //  val kvsPortDetected = Utils.checkExecutorAvailPort(data, kvsIP)
    val kvsPortDetected = 3000
    val kvsPort = data.sparkContext.getConf.getInt("spark.oap.mllib.oneccl.kvs.port",
      kvsPortDetected)

    kvsIP + "_" + kvsPort
  }

  // Return index -> (rows, cols) map
  def getPartitionDims(data: RDD[Vector]): Map[Int, (Int, Int)] = {
    var numCols: Int = 0
    // Collect the numRows and numCols
    val collected = data.mapPartitionsWithIndex { (index: Int, it: Iterator[Vector]) =>
      if (it.hasNext) {
        numCols = it.next().size
        Iterator((index, it.size + 1, numCols))
      } else {
        Iterator((index, 0, numCols))
      }
    }.collect

    var ret = Map[Int, (Int, Int)]()

    // set numRows and numCols
    collected.foreach {
      case (index, rows, cols) =>
        ret += (index -> (rows, cols))
    }

    ret
  }

  def sparkExecutorCores(): Int = {
    val conf = new SparkConf(true)

    // Use 1 if not set
    val executorCores = conf.getInt("spark.executor.cores", 1)

    executorCores
  }

  def sparkFirstExecutorIP(sc: SparkContext): String = {
    // Create empty partitions to start executors
    sc.parallelize(Seq[Int]()).count()

    val info = sc.statusTracker.getExecutorInfos
    // get first executor, info(0) is driver

    val host = if (sc.master.startsWith("local")) {
      info(0).host()
    } else {
      info(1).host()
    }
    val ip = InetAddress.getByName(host).getHostAddress
    ip
  }

  def checkExecutorAvailPort(data: RDD[_], localIP: String): Int = {

    if (localIP == "127.0.0.1" || localIP == "127.0.1.1") {
      println(s"\nOneCCL: Error: doesn't support loopback IP ${localIP}, " +
        s"please assign IP address to your host.\n")
      System.exit(-1)
    }

    val sc = data.sparkContext
    val result = data.mapPartitions { p =>
      val port = OneCCL.getAvailPort(localIP)
      if (port != -1) {
        Iterator(port)
      } else {
        Iterator()
      }
    }.collect()

    result(0)
  }

  //
  // This function is first called in driver and executors to load libraries and check compatibility
  // All other functions using native libraries will depend on this function to be called first
  //
  def checkClusterPlatformCompatibility(sc: SparkContext): Boolean = {
    // check driver platform compatibility
    if (!OneDAL.cCheckPlatformCompatibility()) {
      return false
    }

    // check workers' platform compatibility
    val executor_num = Utils.sparkExecutorNum(sc)
    val data = sc.parallelize(1 to executor_num, executor_num)
    val result = data.mapPartitions { p =>
      Iterator(OneDAL.cCheckPlatformCompatibility())
    }.collect()

    result.forall(_ == true)
  }

  def sparkExecutorNum(sc: SparkContext): Int = {

    if (sc.master.contains("local")) {
      return 1
    }

    // Create empty partitions to start executors
    sc.parallelize(Seq[Int]()).count()

    // Get running executors infos
    val executorInfos = sc.statusTracker.getExecutorInfos

    // Return executor number (exclude driver)
    executorInfos.length - 1
  }
  def getSparkVersion(): String = {
    // For example: CHD spark version is 3.1.1.3.1.7290.5-2.
    // The string before the third dot is the spark version.
    val array = SPARK_VERSION.split("\\.")
    val sparkVersion = if (array.size > 3) {
      val version = array.take(3).mkString(".")
      version
    } else {
      SPARK_VERSION
    }
    sparkVersion
  }
}
