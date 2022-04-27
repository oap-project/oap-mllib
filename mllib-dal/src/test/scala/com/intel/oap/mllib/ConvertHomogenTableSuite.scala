package com.intel.oap.mllib

import com.intel.oneapi.dal.table.Common
import com.intel.oneapi.dal.table.Common.DataLayout.ROWMAJOR
import com.intel.oneapi.dal.table.Common.DataType.FLOAT64
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}

class ConvertHomogenTableSuite extends FunctionsSuite with Logging {

  test("test convert vector to homogentable") {
    val data = Array(
      Vectors.dense(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562),
      Vectors.dense(7.279464,0.390664,-9.619284,3.435376,-4.769490,-4.873188,-0.118791,-5.117316,-0.418655,-0.475422),
      Vectors.dense(-6.615791,-6.191542,0.402459,-9.743521,-9.990568,9.105346,1.691312,-2.605659,9.534952,-7.829027),
      Vectors.dense(-4.792007,-2.491098,-2.939393,8.086467,3.773812,-9.997300,0.222378,8.995244,-5.753282,6.091060),
      Vectors.dense(7.700725,-6.414918,1.684476,-8.983361,4.284580,-9.017608,0.552379,-7.705741,2.589852,0.411561),
      Vectors.dense(6.991900,-1.063721,9.321163,-0.429719,-2.167696,-1.736425,-0.919139,6.980681,-0.711914,3.414347),
      Vectors.dense(5.794488,-1.062261,0.955322,0.389642,3.012921,-9.953994,-3.197309,3.992421,-6.935902,8.147622),
      Vectors.dense(-2.486670,6.973242,-4.047004,-5.655629,5.081786,5.533859,7.821403,2.763313,-0.454056,6.554309),
      Vectors.dense(2.204855,7.839522,7.381886,1.618749,-6.566877,7.584285,-8.355983,-5.501410,-8.191205,-2.608499),
      Vectors.dense(-9.948613,-8.941953,-8.106389,4.863542,5.852806,-1.659259,6.342504,-8.190106,-3.110330,-7.484658),
    )

    val table = OneDAL.makeHomogenTable(data, getDevice)
    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 10)

    assert(table.getDataLayout == ROWMAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == Common.FeatureType.RATIO)
    }

    assert(table.getDoubleData === convertArray(data))

  }

  test("test convert doublearray to homogentable") {
    val data = Array(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562)
    val table = OneDAL.doubleArrayToHomogenTable(data, getDevice)
    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 1)

    assert(table.getDataLayout == ROWMAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == Common.FeatureType.RATIO)
    }
    assert(table.getDoubleData === data)

  }

  test("test convert old vector to homogentable") {
    val data = Array(
      OldVectors.dense(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562),
      OldVectors.dense(7.279464,0.390664,-9.619284,3.435376,-4.769490,-4.873188,-0.118791,-5.117316,-0.418655,-0.475422),
      OldVectors.dense(-6.615791,-6.191542,0.402459,-9.743521,-9.990568,9.105346,1.691312,-2.605659,9.534952,-7.829027),
      OldVectors.dense(-4.792007,-2.491098,-2.939393,8.086467,3.773812,-9.997300,0.222378,8.995244,-5.753282,6.091060),
      OldVectors.dense(7.700725,-6.414918,1.684476,-8.983361,4.284580,-9.017608,0.552379,-7.705741,2.589852,0.411561),
      OldVectors.dense(6.991900,-1.063721,9.321163,-0.429719,-2.167696,-1.736425,-0.919139,6.980681,-0.711914,3.414347),
      OldVectors.dense(5.794488,-1.062261,0.955322,0.389642,3.012921,-9.953994,-3.197309,3.992421,-6.935902,8.147622),
      OldVectors.dense(-2.486670,6.973242,-4.047004,-5.655629,5.081786,5.533859,7.821403,2.763313,-0.454056,6.554309),
      OldVectors.dense(2.204855,7.839522,7.381886,1.618749,-6.566877,7.584285,-8.355983,-5.501410,-8.191205,-2.608499),
      OldVectors.dense(-9.948613,-8.941953,-8.106389,4.863542,5.852806,-1.659259,6.342504,-8.190106,-3.110330,-7.484658),
    )

    val table = OneDAL.makeHomogenTable(data, getDevice)
    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 10)

    assert(table.getDataLayout == ROWMAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == Common.FeatureType.RATIO)
    }

    assert(table.getDoubleData === convertArray(data))
  }

  def convertArray(arrayVectors: Array[Vector]): Array[Double] = {
    val numCols = arrayVectors.head.size
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
    val numCols = arrayVectors.head.size
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

  private def getDevice = {
    val device = System.getProperty("computeDevice")
    var computeDevice: Common.ComputeDevice = Common.ComputeDevice.CPU
    System.out.println("getDevice : " + device)
    if(device != null) {
      device.toUpperCase match {
        case "HOST" =>  computeDevice = Common.ComputeDevice.HOST
        case "CPU"  => computeDevice = Common.ComputeDevice.CPU
        case "GPU"  => computeDevice = Common.ComputeDevice.GPU
        case _  => "Invalid Device"
      }
    }
    System.out.println("getDevice : " + computeDevice)
    computeDevice
  }
}
