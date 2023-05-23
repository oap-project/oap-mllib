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

package org.apache.spark.ml.regression

import org.apache.spark.{SparkConf, TestCommon}

import scala.collection.JavaConverters._
import scala.util.Random
import org.dmg.pmml.{OpType, PMML}
import org.dmg.pmml.regression.{RegressionModel => PMMLRegressionModel}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{ParamMap, ParamsSuite}
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.sql.{DataFrame, Row}


class MLlibLinearRegressionSuite extends MLTest with DefaultReadWriteTest with PMMLReadWriteTest {

  import testImplicits._
  override def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.oap.mllib.device", TestCommon.getComputeDevice.toString)
  }

  private val seed: Int = 42
  @transient var datasetWithDenseFeature: DataFrame = _
  @transient var datasetWithStrongNoise: DataFrame = _
  @transient var datasetWithDenseFeatureWithoutIntercept: DataFrame = _
  @transient var datasetWithSparseFeature: DataFrame = _
  @transient var datasetWithWeight: DataFrame = _
  @transient var datasetWithWeightConstantLabel: DataFrame = _
  @transient var datasetWithWeightZeroLabel: DataFrame = _
  @transient var datasetWithOutlier: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    datasetWithDenseFeature = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 10000, seed, eps = 0.1), 2).map(_.asML).toDF()

    datasetWithStrongNoise = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 100, seed, eps = 5.0), 2).map(_.asML).toDF()

    /*
       datasetWithDenseFeatureWithoutIntercept is not needed for correctness testing
       but is useful for illustrating training model without intercept
     */
    datasetWithDenseFeatureWithoutIntercept = sc.parallelize(
      LinearDataGenerator.generateLinearInput(
        intercept = 0.0, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
        xVariance = Array(0.7, 1.2), nPoints = 10000, seed, eps = 0.1), 2).map(_.asML).toDF()

    val r = new Random(seed)
    // When feature size is larger than 4096, normal optimizer is chosen
    // as the solver of linear regression in the case of "auto" mode.
    val featureSize = 4100
    datasetWithSparseFeature = sc.parallelize(LinearDataGenerator.generateLinearInput(
        intercept = 0.0, weights = Seq.fill(featureSize)(r.nextDouble()).toArray,
        xMean = Seq.fill(featureSize)(r.nextDouble()).toArray,
        xVariance = Seq.fill(featureSize)(r.nextDouble()).toArray, nPoints = 200,
        seed, eps = 0.1, sparsity = 0.7), 2).map(_.asML).toDF()

    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b <- c(17, 19, 23, 29)
       w <- c(1, 2, 3, 4)
       df <- as.data.frame(cbind(A, b))
     */
    datasetWithWeight = sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(19.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(23.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(29.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2).toDF()

    /*
       R code:

       A <- matrix(c(0, 1, 2, 3, 5, 7, 11, 13), 4, 2)
       b.const <- c(17, 17, 17, 17)
       w <- c(1, 2, 3, 4)
       df.const.label <- as.data.frame(cbind(A, b.const))
     */
    datasetWithWeightConstantLabel = sc.parallelize(Seq(
      Instance(17.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(17.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(17.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(17.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2).toDF()

    datasetWithWeightZeroLabel = sc.parallelize(Seq(
      Instance(0.0, 1.0, Vectors.dense(0.0, 5.0).toSparse),
      Instance(0.0, 2.0, Vectors.dense(1.0, 7.0)),
      Instance(0.0, 3.0, Vectors.dense(2.0, 11.0)),
      Instance(0.0, 4.0, Vectors.dense(3.0, 13.0))
    ), 2).toDF()

    datasetWithOutlier = {
      val inlierData = LinearDataGenerator.generateLinearInput(
        intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
        xVariance = Array(0.7, 1.2), nPoints = 900, seed, eps = 0.1)
      val outlierData = LinearDataGenerator.generateLinearInput(
        intercept = -2.1, weights = Array(0.6, -1.2), xMean = Array(0.9, -1.3),
        xVariance = Array(1.5, 0.8), nPoints = 100, seed, eps = 0.1)
      sc.parallelize(inlierData ++ outlierData, 2).map(_.asML).toDF()
    }
  }

  /**
   * Enable the ignored test to export the dataset into CSV format,
   * so we can validate the training accuracy compared with R's glmnet package.
   */
  ignore("export test data into CSV format") {
    datasetWithDenseFeature.rdd.map { case Row(label: Double, features: Vector) =>
      label + "," + features.toArray.mkString(",")
    }.repartition(1).saveAsTextFile("target/tmp/LinearRegressionSuite/datasetWithDenseFeature")

    datasetWithDenseFeatureWithoutIntercept.rdd.map {
      case Row(label: Double, features: Vector) =>
        label + "," + features.toArray.mkString(",")
    }.repartition(1).saveAsTextFile(
      "target/tmp/LinearRegressionSuite/datasetWithDenseFeatureWithoutIntercept")

    datasetWithSparseFeature.rdd.map { case Row(label: Double, features: Vector) =>
      label + "," + features.toArray.mkString(",")
    }.repartition(1).saveAsTextFile("target/tmp/LinearRegressionSuite/datasetWithSparseFeature")

    datasetWithOutlier.rdd.map { case Row(label: Double, features: Vector) =>
      label + "," + features.toArray.mkString(",")
    }.repartition(1).saveAsTextFile("target/tmp/LinearRegressionSuite/datasetWithOutlier")
  }

  // OAP-MLlib we are not supported singular matrices
  // 
  // Ridge regression get wrong ans in CPU
  test("regularized linear regression through origin with constant label") {
    // The problem is ill-defined if fitIntercept=false, regParam is non-zero.
    // An exception is thrown in this case.
    Seq("auto", "l-bfgs", "normal").foreach { solver =>
      for (standardization <- Seq(false, true)) {
        val model = new LinearRegression().setFitIntercept(false)
          .setRegParam(0.1).setStandardization(standardization).setSolver(solver)
        intercept[IllegalArgumentException] {
          model.fit(datasetWithWeightConstantLabel)
        }
      }
    }
  }

  test("linear regression model training summary") {
    Seq("auto", "l-bfgs", "normal").foreach { solver =>
      val trainer = new LinearRegression().setSolver(solver).setPredictionCol("myPrediction")
      val model = trainer.fit(datasetWithDenseFeature)
      val trainerNoPredictionCol = trainer.setPredictionCol("")
      val modelNoPredictionCol = trainerNoPredictionCol.fit(datasetWithDenseFeature)

      // Training results for the model should be available
      assert(model.hasSummary)
      assert(modelNoPredictionCol.hasSummary)

      // Schema should be a superset of the input dataset
      assert((datasetWithDenseFeature.schema.fieldNames.toSet + model.getPredictionCol).subsetOf(
        model.summary.predictions.schema.fieldNames.toSet))
      // Validate that we re-insert a prediction column for evaluation
      val modelNoPredictionColFieldNames
      = modelNoPredictionCol.summary.predictions.schema.fieldNames
      assert(datasetWithDenseFeature.schema.fieldNames.toSet.subsetOf(
        modelNoPredictionColFieldNames.toSet))
      assert(modelNoPredictionColFieldNames.exists(s => s.startsWith("prediction_")))

      // Residuals in [[LinearRegressionResults]] should equal those manually computed
      datasetWithDenseFeature.select("features", "label")
        .rdd
        .map { case Row(features: DenseVector, label: Double) =>
          val prediction =
            features(0) * model.coefficients(0) + features(1) * model.coefficients(1) +
              model.intercept
          label - prediction
        }
        .zip(model.summary.residuals.rdd.map(_.getDouble(0)))
        .collect()
        .foreach { case (manualResidual: Double, resultResidual: Double) =>
          assert(manualResidual ~== resultResidual relTol 1E-5)
        }

      /*
         # Use the following R code to generate model training results.

         # path/part-00000 is the file generated by running LinearDataGenerator.generateLinearInput
         # as described before the beforeAll() method.
         d1 <- read.csv("path/part-00000", header=FALSE, stringsAsFactors=FALSE)
         fit <- glm(V1 ~ V2 + V3, data = d1, family = "gaussian")
         names(f1)[1] = c("V2")
         names(f1)[2] = c("V3")
         f1 <- data.frame(as.numeric(d1$V2), as.numeric(d1$V3))
         predictions <- predict(fit, newdata=f1)
         l1 <- as.numeric(d1$V1)

         residuals <- l1 - predictions
         > mean(residuals^2)           # MSE
         [1] 0.00985449
         > mean(abs(residuals))        # MAD
         [1] 0.07961668
         > cor(predictions, l1)^2   # r^2
         [1] 0.9998737

         > summary(fit)

          Call:
          glm(formula = V1 ~ V2 + V3, family = "gaussian", data = d1)

          Deviance Residuals:
               Min        1Q    Median        3Q       Max
          -0.47082  -0.06797   0.00002   0.06725   0.34635

          Coefficients:
                       Estimate Std. Error t value Pr(>|t|)
          (Intercept) 6.3022157  0.0018600    3388   <2e-16 ***
          V2          4.6982442  0.0011805    3980   <2e-16 ***
          V3          7.1994344  0.0009044    7961   <2e-16 ***

          # R code for r2adj
          lm_fit <- lm(V1 ~ V2 + V3, data = d1)
          summary(lm_fit)$adj.r.squared
          [1] 0.9998736
          ---

          ....
       */
      assert(model.summary.meanSquaredError ~== 0.00985449 relTol 1E-4)
      assert(model.summary.meanAbsoluteError ~== 0.07961668 relTol 1E-4)
      assert(model.summary.r2 ~== 0.9998737 relTol 1E-4)
      assert(model.summary.r2adj ~== 0.9998736  relTol 1E-4)

      // Normal solver uses "WeightedLeastSquares". If no regularization is applied or only L2
      // regularization is applied, this algorithm uses a direct solver and does not generate an
      // objective history because it does not run through iterations.
      if (solver == "l-bfgs") {
        // Objective function should be monotonically decreasing for linear regression
        assert(
          model.summary
            .objectiveHistory
            .sliding(2)
            .forall(x => x(0) >= x(1)))
      } else {
        // To clarify that the normal solver is used here.
        assert(model.summary.objectiveHistory.length == 1)
        assert(model.summary.objectiveHistory(0) == 0.0)
        val devianceResidualsR = Array(-0.47082, 0.34635)
        val seCoefR = Array(0.0011805, 0.0009044, 0.0018600)
        val tValsR = Array(3980, 7961, 3388)
        val pValsR = Array(0, 0, 0)
        model.summary.devianceResiduals.zip(devianceResidualsR).foreach { x =>
          assert(x._1 ~== x._2 absTol 1E-4) }
        model.summary.coefficientStandardErrors.zip(seCoefR).foreach { x =>
          assert(x._1 ~== x._2 absTol 1E-4) }
        model.summary.tValues.map(_.round).zip(tValsR).foreach{ x => assert(x._1 === x._2) }
        model.summary.pValues.map(_.round).zip(pValsR).foreach{ x => assert(x._1 === x._2) }
      }
    }
  }

  test("linear regression with weighted samples") {
    val sqlContext = spark.sqlContext
    import sqlContext.implicits._
    val numClasses = 0
    def modelEquals(m1: LinearRegressionModel, m2: LinearRegressionModel): Unit = {
      assert(m1.coefficients ~== m2.coefficients relTol 0.01)
      assert(m1.intercept ~== m2.intercept relTol 0.01)
    }
    val testParams = Seq(
      // (elasticNetParam, regParam, fitIntercept, standardization)
      (0.0, 0.21, true, true),
      (0.0, 0.21, true, false),
      (0.0, 0.21, false, false),
      (1.0, 0.21, true, true)
    )

    // For squaredError loss
    for (solver <- Seq("auto", "l-bfgs", "normal");
         (elasticNetParam, regParam, fitIntercept, standardization) <- testParams) {
      val estimator = new LinearRegression()
        .setFitIntercept(fitIntercept)
        .setStandardization(standardization)
        .setRegParam(regParam)
        .setElasticNetParam(elasticNetParam)
        .setSolver(solver)
        .setMaxIter(1)
      MLTestingUtils.testArbitrarilyScaledWeights[LinearRegressionModel, LinearRegression](
        datasetWithStrongNoise.as[LabeledPoint], estimator, modelEquals)
      MLTestingUtils.testOutliersWithSmallWeights[LinearRegressionModel, LinearRegression](
        datasetWithStrongNoise.as[LabeledPoint], estimator, numClasses, modelEquals,
        outlierRatio = 3)
      MLTestingUtils.testOversamplingVsWeighting[LinearRegressionModel, LinearRegression](
        datasetWithStrongNoise.as[LabeledPoint], estimator, modelEquals, seed)
    }

    // For huber loss
    for ((_, regParam, fitIntercept, standardization) <- testParams) {
      val estimator = new LinearRegression()
        .setLoss("huber")
        .setFitIntercept(fitIntercept)
        .setStandardization(standardization)
        .setRegParam(regParam)
        .setMaxIter(1)
      MLTestingUtils.testArbitrarilyScaledWeights[LinearRegressionModel, LinearRegression](
        datasetWithOutlier.as[LabeledPoint], estimator, modelEquals)
      MLTestingUtils.testOutliersWithSmallWeights[LinearRegressionModel, LinearRegression](
        datasetWithOutlier.as[LabeledPoint], estimator, numClasses, modelEquals,
        outlierRatio = 3)
      MLTestingUtils.testOversamplingVsWeighting[LinearRegressionModel, LinearRegression](
        datasetWithOutlier.as[LabeledPoint], estimator, modelEquals, seed)
    }
  }

}

object LinearRegressionSuite {

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allParamSettings: Map[String, Any] = Map(
    "predictionCol" -> "myPrediction",
    "regParam" -> 0.01,
    "elasticNetParam" -> 0.1,
    "maxIter" -> 2,  // intentionally small
    "fitIntercept" -> true,
    "tol" -> 0.8,
    "standardization" -> false,
    "solver" -> "l-bfgs"
  )
}
