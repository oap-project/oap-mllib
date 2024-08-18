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
package com.intel.oap.mllib

import com.intel.oneapi.dal.table.{Common, HomogenTable, RowAccessor}
import org.apache.spark.ml.linalg.{DenseMatrix, Vector}
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
  def getMatrixFromTable(table: HomogenTable): DenseMatrix = {
    val numRows = table.getRowCount.toInt
    val numCols = table.getColumnCount.toInt
    // returned DoubleBuffer is ByteByffer, need to copy as double array
    val accessor = new RowAccessor(table.getcObejct())
    val arrayDouble: Array[Double] = accessor.pullDouble(0, numRows)

    // Transpose as DAL numeric table is row-major and DenseMatrix is column major
    val matrix = new DenseMatrix(numRows, numCols, arrayDouble, isTransposed = true)

    matrix
  }

  def getComputeDevice: CommonJob.ComputeDevice = {
    val device = System.getProperty("computeDevice")
    var computeDevice: CommonJob.ComputeDevice = CommonJob.ComputeDevice.HOST
    if(device != null) {
      device.toUpperCase match {
        case "HOST" => computeDevice = CommonJob.ComputeDevice.HOST
        case "CPU" => computeDevice = CommonJob.ComputeDevice.CPU
        case "GPU" => computeDevice = CommonJob.ComputeDevice.GPU
        case _ => "Invalid Device"
      }
    }
    System.out.println("getDevice : " + computeDevice)
    computeDevice
  }
}
