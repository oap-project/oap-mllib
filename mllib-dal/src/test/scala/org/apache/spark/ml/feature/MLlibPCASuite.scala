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

package org.apache.spark.ml.feature

import org.apache.spark.{SparkConf, TestCommon}
import org.apache.spark.ml.linalg.{DenseMatrix, DenseVector, Matrices, Vector, Vectors}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.sql.Row

class MLlibPCASuite extends MLTest with DefaultReadWriteTest {

  import testImplicits._
  override def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.oap.mllib.device", TestCommon.getComputeDevice.toString)
    conf.set("spark.driver.bindAddress", "10.239.34.1");

  }

  test("params") {
    ParamsSuite.checkParams(new PCA)
    val mat = Matrices.dense(2, 2, Array(0.0, 1.0, 2.0, 3.0)).asInstanceOf[DenseMatrix]
    val explainedVariance = Vectors.dense(0.5, 0.5).asInstanceOf[DenseVector]
    val model = new PCAModel("pca", mat, explainedVariance)
    ParamsSuite.checkParams(model)
  }

  // This case is modified for sign flip and small explained variance
  test("pca") {
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0),
    )

    val k = 3

    val dataRDD = sc.parallelize(data, 2)

    val mat = new RowMatrix(dataRDD.map(OldVectors.fromML))
    val result = mat.computePrincipalComponentsAndExplainedVariance(k)
    val expectedPC = result._1
    val expectedVariance = result._2

    println("Spark MLlib: ")
    println(s"Principle Components: \n${expectedPC}")
    println(s"Explained Variance: \n${expectedVariance}")

    val df = dataRDD.map(Tuple1(_)).toDF("features")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pca_features")
      .setK(k)

    val pcaModel = pca.fit(df)

    println("OAP MLlib: ")
    println(s"Principle Components: \n${pcaModel.pc}")
    println(s"Explained Variance: \n${pcaModel.explainedVariance}")

    assert(pcaModel.explainedVariance ~== expectedVariance.asML absTol 1e-5,
      "Explained variance result is different with expected one.")

    val expectedCol = expectedPC.asML.colIter
    val resultCol = pcaModel.pc.colIter

    for ((elem, index) <- resultCol.zip(expectedCol).zipWithIndex) {
      // use absolute value to compare due to sign flip
      val resultVec = Vectors.dense(elem._1.toArray.map(math.abs(_)))
      val expectedVec = Vectors.dense(elem._2.toArray.map(math.abs(_)))

      // Compare pc only for large enough explained variance
      if (expectedVariance(index) > 1e-5)
        assert(resultVec ~== expectedVec absTol 1e-5,
          "Principle components result (absolute value) is different with expected one.")
        println(expectedVariance(index))
    }
  }

  test("PCA read/write") {
    val t = new PCA()
      .setInputCol("myInputCol")
      .setOutputCol("myOutputCol")
      .setK(3)
    testDefaultReadWrite(t)
  }

  test("PCAModel read/write") {
    val instance = new PCAModel("myPCAModel",
      Matrices.dense(2, 2, Array(0.0, 1.0, 2.0, 3.0)).asInstanceOf[DenseMatrix],
      Vectors.dense(0.5, 0.5).asInstanceOf[DenseVector])
    val newInstance = testDefaultReadWrite(instance)
    assert(newInstance.pc === instance.pc)
  }

  test("normalized data") {
    val norData = Seq(
      Vectors.dense(-2.0,0.6666666666666667,-1.0,1.333333333333333,-4.0),
      Vectors.dense(0.0,-0.3333333333333333,2.0,-1.666666666666667,1.0),
      Vectors.dense(2.0,-0.3333333333333333,-1.0,0.33333333333333304,3.0),
    )

    val k = 2
    val nordataRDD = sc.parallelize(norData, 2)

    val mat = new RowMatrix(nordataRDD.map(OldVectors.fromML))
    val pc = mat.computePrincipalComponents(k)
    val expected = mat.multiply(pc).rows.map(_.asML)

    val df = nordataRDD.zip(expected).toDF("features", "expected")

    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(k)

    val training_df = df.select("features")
    val pcaModel = pca.fit(training_df)
    val transformed = pcaModel.transform(training_df)
    transformed.show(false)
    checkVectorSizeOnDF(transformed, "pcaFeatures", pcaModel.getK)

    MLTestingUtils.checkCopyAndUids(pca, pcaModel)
    testTransformer[(Vector, Vector)](df, pcaModel, "pcaFeatures", "expected") {
      case Row(result: Vector, expected: Vector) =>
        val newResultABS = result.toArray.map(f => f.abs)
        val newExpectedABS = expected.toArray.map(f => f.abs )
        val newResult = Vectors.dense(newResultABS)
        val newExpected = Vectors.dense(newExpectedABS)
        assert(newResult ~== newExpected absTol 1e-5,
          "Transformed vector is different with expected vector.")
    }
  }

  test("unnormalized data") {
    val data = Array(
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )
    val k = 2

    val dataRDD = sc.parallelize(data, 2)

    val mat = new RowMatrix(dataRDD.map(OldVectors.fromML))
    val pc = mat.computePrincipalComponents(k)
    val expected = mat.multiply(pc).rows.map(_.asML)

    val df = dataRDD.zip(expected).toDF("features", "expected")
    df.show(false)
    val training_df = df.select("features")
    val pca = new PCA()
      .setInputCol("features")
      .setOutputCol("pcaFeatures")
      .setK(k)

    val pcaModel = pca.fit(training_df)
    val transformed = pcaModel.transform(training_df)
    transformed.show(false)
    checkVectorSizeOnDF(transformed, "pcaFeatures", pcaModel.getK)

    MLTestingUtils.checkCopyAndUids(pca, pcaModel)
    testTransformer[(Vector, Vector)](df, pcaModel, "pcaFeatures", "expected") {
      case Row(result: Vector, expected: Vector) =>
        val newResultABS = result.toArray.map(f => f.abs)
        val newExpectedABS = expected.toArray.map(f => f.abs )
        val newResult = Vectors.dense(newResultABS)
        val newExpected = Vectors.dense(newExpectedABS)
        assert(newResult ~== newExpected absTol 1e-5,
          "Transformed vector is different with expected vector.")
    }
  }
}
