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
import java.io.{File, FileWriter, BufferedWriter}
import java.time.LocalDateTime
import java.time.Duration
import java.time.format.DateTimeFormatter
import collection.mutable.ListBuffer

object Tabulator {
  def format(table: Seq[Seq[Any]]) = table match {
    case Seq() => ""
    case _ => 
    val sizes = for (row <- table) yield (for (cell <- row) yield if (cell == null) 0 else cell.toString.length)
    val colSizes = for (col <- sizes.transpose) yield col.max
    val rows = for (row <- table) yield formatRow(row, colSizes)
    formatRows(rowSeparator(colSizes), rows)
    }

  def formatRows(rowSeparator: String, rows: Seq[String]): String = (
    rowSeparator :: 
    rows.head :: 
    rowSeparator :: 
    rows.tail.toList ::: 
    rowSeparator :: 
    List()).mkString("\n")

  def formatRow(row: Seq[Any], colSizes: Seq[Int]) = {
    val cells = (for ((item, size) <- row.zip(colSizes)) yield if (size == 0) "" else ("%" + size + "s").format(item))
      cells.mkString("|", "|", "|")
    }

  def rowSeparator(colSizes: Seq[Int]) = colSizes map { "-" * _ } mkString("+", "+", "+")
}



object Utils {

  def calculateSingle(pre: Long, cur: List[Long]): List[String] = {
    if (cur.tail.isEmpty) {
      return List((cur.head - pre).toString())
    }
    return List((cur.head - pre).toString()) ++ calculateSingle(cur.head, cur.tail)
  }

  def calculateTimeZone(raw: List[Long]): List[String] = {
    return List(raw.head.toString()) ++ calculateSingle(raw.head, raw.tail)
  }

  class AlgoTimeStamp(var name: String) {
    var timeStamp = LocalDateTime.now()
    var timeStampHuman = "uninitialized time stamp"
    def update(timeFileName: String): Unit = {
      timeStamp = LocalDateTime.now()
      timeStampHuman = DateTimeFormatter.ofPattern("yyyy-M-dd HH:mm:ss.SSS").format(timeStamp)
      val timeFile = new BufferedWriter(new FileWriter(new File(timeFileName), true))
      timeFile.write(name + "," + timeStampHuman + "\n")
      timeFile.close
    }
  }

  class AlgoTimeMetrics(val algoName: String) {
    val timeZoneName = List("Preprocessing", "Data Convertion", "Training")
    val algoTimeStampList = timeZoneName.map((x: String) => (x, new Utils.AlgoTimeStamp(x))).toMap
    val recorderName = Utils.GlobalTimeTable.register(this)
    val timeFileName = recorderName + "time_breakdown"

    val timerEnabled = isTimerEnabled()

    def record(stampName: String): Unit = {
      if (timerEnabled) {
        algoTimeStampList(stampName).update(timeFileName)
      }
    }

    def getTableHead(): List[String] = {
      List("AlgoName", "Start Time") ++ timeZoneName.tail.map(x => x + "(ms)") ++ List("Total time(ms)")
    }
    def getTableContent(): List[String] = {
      val (startTimeStampName, startTime) = algoTimeStampList.head
      val (endTimeStampName, endTime) = algoTimeStampList.last
      val contentMain = algoTimeStampList.view.map{case(k, v) => (Duration.between(startTime.timeStamp, v.timeStamp).toMillis())}.toList.tail
      List(recorderName) ++ List(startTime.timeStampHuman) ++ calculateTimeZone(contentMain) ++ List((Duration.between(startTime.timeStamp, endTime.timeStamp).toMillis()).toString())
    }

    def print(): Unit = {
      if (timerEnabled) {
        println("OAP MLlib: time metrics")
        println(Tabulator.format(List(getTableHead ,getTableContent)))
      }
    }

    def writeToFile(filename: String): Unit = {
    }
  }

  class TimeMetricsTable {
    private var _algoTimeMetricsList = collection.mutable.Map[String, collection.mutable.ListBuffer[AlgoTimeMetrics]]()
    def register(timeMetrics: AlgoTimeMetrics): String = {
      val timeClassList = _algoTimeMetricsList.getOrElseUpdate(timeMetrics.algoName, ListBuffer())
      timeClassList += timeMetrics
      val recorderName = timeMetrics.algoName + timeClassList.size.toString
      return recorderName
    }
  }


  val GlobalTimeTable = new TimeMetricsTable()

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

  def isTimerEnabled(): Boolean = {
    val sc = SparkSession.active.sparkContext
    sc.getConf.getBoolean("spark.oap.mllib.timer", true)
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
