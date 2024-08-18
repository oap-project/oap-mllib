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

import org.junit.jupiter.api.Assertions.assertArrayEquals

import com.intel.daal.data_management.data.{Matrix => DALMatrix, NumericTable}
import com.intel.daal.services.DaalContext
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import com.intel.oneapi.dal.table.Common.DataLayout.ROW_MAJOR

import com.intel.oneapi.dal.table.Common.DataType.FLOAT64
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.{DenseMatrix, Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Matrix => OldMatrix, Vector => OldVector}


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

    val table = OneDAL.makeHomogenTable(data, TestCommon.getComputeDevice)

    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 10)

    assert(table.getDataLayout == ROW_MAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == CommonJob.FeatureType.RATIO)
    }

    assertArrayEquals(table.getDoubleData, TestCommon.convertArray(data))
  }

  test("test convert doublearray to homogentable") {
    val data = Array(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562)
    val table = OneDAL.doubleArrayToHomogenTable(data, TestCommon.getComputeDevice)

    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 1)

    assert(table.getDataLayout == ROW_MAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == CommonJob.FeatureType.RATIO)
    }
    assertArrayEquals(table.getDoubleData, data)

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

    val table = OneDAL.makeHomogenTable(data, TestCommon.getComputeDevice)

    assert(table.hasData == true)
    assert(table.getColumnCount == 10)
    assert(table.getRowCount == 10)

    assert(table.getDataLayout == ROW_MAJOR)
    val metadata = table.getMetaData
    for (i <- 0 until 10) {
      assert(metadata.getDataType(i) == FLOAT64)
      assert(metadata.getFeatureType(i) == CommonJob.FeatureType.RATIO)
    }

    assertArrayEquals(table.getDoubleData, TestCommon.convertArray(data))
  }

  test("test convert homogentable 1xN to vector "){
    val data = Array(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562)

    val expectData = Array(5.308206,9.869278)
    val table = new HomogenTable(5, 2, data, TestCommon.getComputeDevice)
    val vector = OneDAL.homogenTable1xNToVector(table)

    assertArrayEquals(expectData, vector.toArray)
  }

  test("test convert numerictable Nx1 to vector "){
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
    val expectData = Array(5.308206, 7.279464, -6.615791, -4.792007, 7.700725, 6.991900, 5.794488, -2.486670, 2.204855, -9.948613)
    val matrix = OneDAL.makeNumericTable(data)
    val vector = OneDAL.numericTableNx1ToVector(matrix)
    assertArrayEquals(expectData, vector.toArray)
  }

  test("test convert homogentable Nx1 to vector "){
    val data = Array(5.236359d, 8.718667d,
                     40.724176d, 10.770023d,
                     90.119887d, 3.815366d,
                     53.620204d, 33.219769d,
                     85.208661d, 15.966239d)
    val expectData = Array(5.236359d, 40.724176d, 90.119887d, 53.620204d, 85.208661d)
    val table = new HomogenTable(5, 2, data, TestCommon.getComputeDevice)
    val vector = OneDAL.homogenTableNx1ToVector(table.getcObejct())

    assertArrayEquals(expectData, vector.toArray)
  }

  test("test convert homogentable to matrix "){
    val data = Array(5.236359d, 8.718667d,
      40.724176d, 10.770023d,
      90.119887d, 3.815366d,
      53.620204d, 33.219769d,
      85.208661d, 15.966239d)
    val expectMatrix = new DenseMatrix(5, 2, data, isTransposed = true)
    val table = new HomogenTable(5, 2, data, TestCommon.getComputeDevice)

    val matrix = OneDAL.homogenTableToMatrix(table)

    assertArrayEquals(expectMatrix.toArray, matrix.toArray)
  }

  test("test convert homogentable to oldmatrix "){
    val data = Array(5.236359d, 8.718667d,
      40.724176d, 10.770023d,
      90.119887d, 3.815366d,
      53.620204d, 33.219769d,
      85.208661d, 15.966239d)
    val expectMatrix = new OldDenseMatrix(5, 2, data, isTransposed = true)
    val table = new HomogenTable(5, 2, data, TestCommon.getComputeDevice)

    val matrix = OneDAL.homogenTableToOldMatrix(table)

    assertArrayEquals(expectMatrix.toArray, matrix.toArray)
  }

  test("test convert homogentable to vector "){
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

    val arrayData = TestCommon.convertArray(data)
    val table = new HomogenTable(10, 10, arrayData, TestCommon.getComputeDevice)
    val array = OneDAL.homogenTableToVectors(table)

    assertArrayEquals(TestCommon.convertArray(data), TestCommon.convertArray(array))
  }
}
