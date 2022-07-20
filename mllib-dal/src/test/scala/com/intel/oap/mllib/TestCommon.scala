package com.intel.oap.mllib

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import com.intel.oneapi.dal.table.{Common, HomogenTable}

import scala.io.Source

object TestCommon {

  def readCSV(path: String): Array[Array[Double]] = {
    val bufferedSource = Source.fromFile(path)
    var matrix: Array[Array[Double]] = Array.empty
    for (line <- bufferedSource.getLines) {
      val cols = line.split(",").map(_.trim.toDouble)
      matrix = matrix :+ cols
    }
    bufferedSource.close
    matrix
  }

  def convertArray(arrayVectors: Array[Array[Double]]): Array[Double] = {
    val numCols: Int = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( array: Array[Double] <- arrayVectors) {
      for (i <- 0 until array.toArray.length ) {
        arrayDouble(index) = array(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    arrayDouble
  }

  def convertArray(arrayVectors: Array[Vector]): Array[Double] = {
    val numCols: Int = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( vector: Vector <- arrayVectors) {
      for (i <- 0 until vector.toArray.length ) {
        arrayDouble(index) = vector(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    arrayDouble
  }

  def convertArray(arrayVectors: Array[OldVector]): Array[Double] = {
    val numCols: Int = arrayVectors.head.size
    val numRows: Int = arrayVectors.size
    val arrayDouble = new Array[Double](numRows * numCols)
    var index = 0
    for( vector: OldVector <- arrayVectors) {
      for (i <- 0 until vector.toArray.length ) {
        arrayDouble(index) = vector(i)
        if (index < (numRows * numCols)) {
          index = index + 1
        }
      }
    }
    arrayDouble
  }

  def getDevice: Common.ComputeDevice = {
    val device = System.getProperty("computeDevice")
    var computeDevice: Common.ComputeDevice = Common.ComputeDevice.HOST
    if(device != null) {
      device.toUpperCase match {
        case "HOST" => computeDevice = Common.ComputeDevice.HOST
        case "CPU"  => computeDevice = Common.ComputeDevice.CPU
        case "GPU"  => computeDevice = Common.ComputeDevice.GPU
        case _  => "Invalid Device"
      }
    }
    System.out.println("getDevice : " + computeDevice)
    computeDevice
  }
}
