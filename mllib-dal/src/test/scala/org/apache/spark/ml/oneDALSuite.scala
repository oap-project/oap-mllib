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
package org.apache.spark.ml

import org.junit.jupiter.api.Assertions.assertArrayEquals
import com.intel.oap.mllib.OneDAL
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import com.intel.oneapi.dal.table.HomogenTable
import org.apache.spark.{SparkContext, TestCommon}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class oneDALSuite extends FunctionsSuite with Logging {

  import testImplicits._

  test("test sparse vector to CSRNumericTable") {
    val data = Seq(
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0), (2, 3.0))),
      Vectors.sparse(3, Seq((0, 10.0), (1, 20.0), (2, 30.0))),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0))),
      Vectors.sparse(3, Seq((0, 10.0), (2, 20.0))),
    )
    val df = data.map(Tuple1.apply).toDF("features")
    df.show()
    val rowsRDD = df.rdd.map {
      case Row(features: Vector) => features
    }
    val results = rowsRDD.coalesce(1).mapPartitions { it: Iterator[Vector] =>
      val vectors: Array[Vector] = it.toArray
      val numColumns = vectors(0).size
      val CSRNumericTable = {
        OneDAL.vectorsToSparseNumericTable(vectors, numColumns)
      }
      Iterator(CSRNumericTable.getCNumericTable)
    }.collect()
    val csr = OneDAL.makeNumericTable(results(0))
    val resultMatrix = OneDAL.numericTableToMatrix(csr)
    val matrix = Matrices.fromVectors(data)

    assertArrayEquals(resultMatrix.toArray, matrix.toArray)
  }

  test("test rddLabeledPoint to merged HomogenTables") {
    val data : RDD[LabeledPoint] = generateLabeledPointRDD(sc, 10, 2, 3)
    val df = data.toDF("label", "features")
    val expect = df.rdd.coalesce(1).mapPartitions {
      it: Iterator[Row] =>
          val featureArray = ArrayBuffer[Double]()
          val labelArray = ArrayBuffer[Double]()
          it.foreach { case Row(label: Double, features: Vector) =>
            val featArray = features.toArray
            for (i <- 0 until featArray.length ) {
              featureArray.append(featArray(i))
            }
            labelArray.append(label)
        }
        Iterator(featureArray.toArray, labelArray.toArray)
    }.collect()
    df.show(10, false)

    val mergedata = OneDAL.coalesceLabelPointsToHomogenTables(df,
      "label", "features", 1, TestCommon.getComputeDevice)
    val results = mergedata.collect()
    val featureTable = new HomogenTable(results(0)._1.split("_")(0).toLong)
    val labelTable = new HomogenTable(results(0)._2.split("_")(0).toLong)

    val fData: Array[Double] = featureTable.getDoubleData()
    val lData: Array[Double] = labelTable.getDoubleData()
    assertArrayEquals(fData, expect(0))
    assertArrayEquals(lData, expect(1))
  }

  test("test rddVector to merged homogenTable") {
    val data = Array(
      Vectors.dense(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562),
      Vectors.dense(7.279464,0.390664,-9.619284,3.435376,-4.769490,-4.873188,-0.118791,-5.117316,-0.418655,-0.475422),
      Vectors.dense(-6.615791,-6.191542,0.402459,-9.743521,-9.990568,9.105346,1.691312,-2.605659,9.534952,-7.829027),
    )
    val expectData = Array(5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,
      -3.256320,-6.029562, 7.279464,0.390664,-9.619284,3.435376,-4.769490,-4.873188,-0.118791,-5.117316,
      -0.418655,-0.475422,-6.615791,-6.191542,0.402459,-9.743521,-9.990568,9.105346,1.691312,-2.605659,
      9.534952,-7.829027)
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    df.show(10 ,false)
    val rddVectors = df.rdd.map {
      case Row(v: Vector) => v
    }.cache()
    val result = OneDAL.coalesceVectorsToHomogenTables(rddVectors, 1, TestCommon.getComputeDevice)

    val tableAddr = result.collect()
    val table = new HomogenTable(tableAddr(0)._1)
    val rData: Array[Double] = table.getDoubleData()
    assertArrayEquals(rData, expectData)
  }

  def generateLabeledPointRDD(
                           sc: SparkContext,
                           nexamples: Int,
                           nfeatures: Int,
                           eps: Double,
                           nparts: Int = 2,
                           probOne: Double = 0.5): RDD[LabeledPoint] = {
    val data = sc.parallelize(0 until nexamples, nparts).map { idx =>
      val rnd = new Random(42 + idx)

      val y = if (idx % 2 == 0) 0.0 else 1.0
      val x = Array.fill[Double](nfeatures) {
        rnd.nextGaussian() + (y * eps)
      }
      LabeledPoint(y, Vectors.dense(x))
    }
    data
  }
}
