package org.apache.spark.ml.util

import java.net.InetAddress

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.Vector

object Utils {

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

  def sparkExecutorNum(sc: SparkContext): Int = {

    if (sc.master.contains("local"))
      return 1

    // Create empty partitions to start executors
    sc.parallelize(Seq[Int]()).count()

    // Get running executors infos
    val executorInfos = sc.statusTracker.getExecutorInfos

    // Return executor number (exclude driver)
    executorInfos.length - 1
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

    val host = if (sc.master.startsWith("local"))
      info(0).host()
    else
      info(1).host()
    val ip = InetAddress.getByName(host).getHostAddress
    ip
  }

  def checkClusterPlatformCompatibility(sc: SparkContext) : Boolean = {
    LibLoader.loadLibMLlibDAL()

    // check driver platform compatibility
    if (!OneDAL.cCheckPlatformCompatibility())
      return false

    // check workers' platform compatibility
    val executor_num = Utils.sparkExecutorNum(sc)
    val data = sc.parallelize(1 to executor_num, executor_num)
    val result = data.map { p =>
      LibLoader.loadLibMLlibDAL()
      OneDAL.cCheckPlatformCompatibility()
    }.collect()

    return result.forall( _ == true)
  }
}
