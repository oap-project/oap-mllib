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

import com.intel.oneapi.dal.table.Common
import org.apache.spark.{SparkConf, SparkContext, SparkFunSuite}
import org.apache.spark.TestUtils.{createTempJsonFile, createTempScriptWithExpectedOutput}
import org.apache.spark.internal.config.Worker.SPARK_WORKER_RESOURCE_FILE
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.resource.ResourceAllocation
import org.apache.spark.resource.TestResourceIDs.{EXECUTOR_GPU_ID, TASK_GPU_ID, WORKER_GPU_ID}
import org.apache.spark.sql.execution.streaming.CommitMetadata.format
import org.apache.spark.sql.test.TestSparkSession
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.util.Utils
import org.json4s.Extraction

/**
 * Test suite for [[RandomForestRegressor]].
 */
class RandomForestRegressorSuite extends MLTest with DefaultReadWriteTest{

  import testImplicits._
  override def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.oap.mllib.device", Common.ComputeDevice.GPU.toString)
    conf.set("spark.oap.mllib.isuite", "true")

    conf
  }

  override def createSparkSession: TestSparkSession = {
    new TestSparkSession(new SparkContext("local[1]", "MLlibUnitTest", sparkConf))
  }

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _
  private var linearRegressionData: DataFrame = _
  private val seed = 42

  override def beforeAll(): Unit = {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000)
        .map(_.asML))

    linearRegressionData = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 1000, seed, eps = 0.5), 2).map(_.asML).toDF()
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("prediction on single instance") {
    val rf = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)

    val df = orderedLabeledPoints50_1000.toDF()
    val model = rf.fit(df)
    testPredictionModelSinglePrediction(model, df)
  }

  test("model support predict leaf index") {
    val model0 = new DecisionTreeRegressionModel("dtc", TreeTests.root0, 3)
    val model1 = new DecisionTreeRegressionModel("dtc", TreeTests.root1, 3)
    val model = new RandomForestRegressionModel("rfr", Array(model0, model1), 3)
    model.setLeafCol("predictedLeafId")
      .setPredictionCol("")

    val data = TreeTests.getTwoTreesLeafData
    data.foreach { case (leafId, vec) => assert(leafId === model.predictLeaf(vec)) }

    val df = sc.parallelize(data, 1).toDF("leafId", "features")
    model.transform(df).select("leafId", "predictedLeafId")
      .collect()
      .foreach { case Row(leafId: Vector, predictedLeafId: Vector) =>
        assert(leafId === predictedLeafId)
      }
  }

  test("should support all NumericType labels and not support other types") {
    val rf = new RandomForestRegressor().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[RandomForestRegressionModel, RandomForestRegressor](
      rf, spark, isClassification = false) { (expected, actual) =>
      TreeTests.checkEqual(expected, actual)
    }
  }

  test("tree params") {
    val rf = new RandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(3)
      .setSeed(123)

    val df = orderedLabeledPoints50_1000.toDF()
    val model = rf.fit(df)
    println(model.getMaxDepth)
    model.trees.foreach (i => {
      assert(i.getMaxDepth === model.getMaxDepth)
      // onedal has bug about set_seed (https://jira.devtools.intel.com/browse/DAALL-6699)
      // assert(i.getSeed === model.getSeed)
      assert(i.getImpurity === model.getImpurity)
      assert(i.getMaxBins === model.getMaxBins)
    })
  }

  test("training with sample weights") {
    val df = linearRegressionData
    val numClasses = 0
    // (numTrees, maxDepth, subsamplingRate, fractionInTol)
    val testParams = Seq(
      (50, 5, 1.0, 0.75),
      (50, 10, 1.0, 0.75),
      (50, 10, 0.95, 0.75)
    )

    for ((numTrees, maxDepth, subsamplingRate, tol) <- testParams) {
      val estimator = new RandomForestRegressor()
        .setNumTrees(numTrees)
        .setMaxDepth(maxDepth)
        .setSubsamplingRate(subsamplingRate)
        .setSeed(seed)
        .setMinWeightFractionPerNode(0.05)

      MLTestingUtils.testArbitrarilyScaledWeights[RandomForestRegressionModel,
        RandomForestRegressor](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, _ ~= _ relTol 0.2, tol))
      MLTestingUtils.testOutliersWithSmallWeights[RandomForestRegressionModel,
        RandomForestRegressor](df.as[LabeledPoint], estimator,
        numClasses, MLTestingUtils.modelPredictionEquals(df, _ ~= _ relTol 0.2, tol),
        outlierRatio = 2)
      MLTestingUtils.testOversamplingVsWeighting[RandomForestRegressionModel,
        RandomForestRegressor](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, _ ~= _ relTol 0.2, tol), seed)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
                        model: RandomForestRegressionModel,
                        model2: RandomForestRegressionModel): Unit = {
      TreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
    }

    val rf = new RandomForestRegressor().setNumTrees(2)
    val rdd = TreeTests.getTreeReadWriteData(sc)

    val allParamSettings = TreeTests.allParamSettings ++ Map("impurity" -> "variance")

    val continuousData: DataFrame =
      TreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 0)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings,
      allParamSettings, checkModelData)
  }

  test("SPARK-33398: Load RandomForestRegressionModel prior to Spark 3.0") {
    val path = testFile("ml-models/rfr-2.4.7")
    val model = RandomForestRegressionModel.load(path)
    assert(model.numFeatures === 692)
    assert(model.totalNumNodes === 8)
    assert(model.trees.map(_.numNodes) === Array(5, 3))

    val metadata = spark.read.json(s"$path/metadata")
    val sparkVersionStr = metadata.select("sparkVersion").first().getString(0)
    assert(sparkVersionStr === "2.4.7")
  }
}

private object RandomForestRegressorSuite extends SparkFunSuite {

  /**
   * Train 2 models on the given dataset, one using the old API and one using the new API.
   * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
                   data: RDD[LabeledPoint],
                   rf: RandomForestRegressor,
                   categoricalFeatures: Map[Int, Int]): Unit = {
    val numFeatures = data.first().features.size
    val oldStrategy =
      rf.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, rf.getOldImpurity)
    val oldModel = OldRandomForest.trainRegressor(data.map(OldLabeledPoint.fromML), oldStrategy,
      rf.getNumTrees, rf.getFeatureSubsetStrategy, rf.getSeed.toInt)
    val newData: DataFrame = TreeTests.setMetadata(data, categoricalFeatures, numClasses = 0)
    val newModel = rf.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    val oldModelAsNew = RandomForestRegressionModel.fromOld(
      oldModel, newModel.parent.asInstanceOf[RandomForestRegressor], categoricalFeatures)
    TreeTests.checkEqual(oldModelAsNew, newModel)
    assert(newModel.numFeatures === numFeatures)
  }
}
