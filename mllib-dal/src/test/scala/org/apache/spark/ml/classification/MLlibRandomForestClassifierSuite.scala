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

package org.apache.spark.ml.classification

import com.intel.oneapi.dal.table.Common

import org.apache.spark.{SparkConf, SparkContext, SparkFunSuite, TaskContext, TestCommon}
import org.apache.spark.ml.classification.LinearSVCSuite.generateSVMInput
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.TreeTests
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.rdd.RDD
import org.apache.spark.resource.TestResourceIDs.{EXECUTOR_GPU_ID, TASK_GPU_ID, WORKER_GPU_ID}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.test.TestSparkSession
import org.apache.spark.util.Utils

/**
 * Test suite for [[RandomForestClassifier]].
 */
class MLlibRandomForestClassifierSuite extends MLTest with DefaultReadWriteTest {

  import RandomForestClassifierSuite.compareAPIs
  import testImplicits._
  override def sparkConf: SparkConf = {
    val conf = super.sparkConf
    conf.set("spark.oap.mllib.device", Common.ComputeDevice.GPU.toString)
    conf.set("spark.oap.mllib.isTest", "true")

    conf
  }

  private var orderedLabeledPoints50_1000: RDD[LabeledPoint] = _
  private var orderedLabeledPoints5_20: RDD[LabeledPoint] = _
  private var binaryDataset: DataFrame = _
  private val seed = 42

  override def beforeAll(): Unit = {
    super.beforeAll()
    orderedLabeledPoints50_1000 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 50, 1000))
        .map(_.asML)
    orderedLabeledPoints5_20 =
      sc.parallelize(EnsembleTestHelper.generateOrderedLabeledPoints(numFeatures = 5, 20))
        .map(_.asML)
    binaryDataset = generateSVMInput(0.01, Array[Double](-1.5, 1.0), 1000, seed).toDF()

  }
  
  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("params") {
    ParamsSuite.checkParams(new RandomForestClassifier)
    val model = new RandomForestClassificationModel("rfc",
      Array(new DecisionTreeClassificationModel("dtc", new LeafNode(0.0, 0.0, null), 1, 2)), 2, 2)
    ParamsSuite.checkParams(model)
  }

  test("predictRaw and predictProbability") {
    val rdd = orderedLabeledPoints5_20
    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = TreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)

    MLTestingUtils.checkCopyAndUids(rf, model)

    testTransformer[(Vector, Double, Double)](df, model, "prediction", "rawPrediction",
      "probability") { case Row(pred: Double, rawPred: Vector, probPred: Vector) =>
      assert(pred === rawPred.argmax,
        s"Expected prediction $pred but calculated ${rawPred.argmax} from rawPrediction.")
      val sum = rawPred.toArray.sum
      assert(Vectors.dense(rawPred.toArray.map(_ / sum)) === probPred,
        "probability prediction mismatch")
      assert(probPred.toArray.sum ~== 1.0 relTol 1E-5)
    }

    ProbabilisticClassifierSuite.testPredictMethods[
      Vector, RandomForestClassificationModel](this, model, df)
  }

  test("prediction on single instance") {
    val rdd = orderedLabeledPoints5_20
    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = TreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)

    testPredictionModelSinglePrediction(model, df)
    testClassificationModelSingleRawPrediction(model, df)
    testProbClassificationModelSingleProbPrediction(model, df)
  }

  test("Fitting without numClasses in metadata") {
    val df: DataFrame = TreeTests.featureImportanceData(sc).toDF()
    val rf = new RandomForestClassifier().setMaxDepth(1).setNumTrees(1)
    rf.fit(df)
  }

  test("model support predict leaf index") {
    val model0 = new DecisionTreeClassificationModel("dtc", TreeTests.root0, 3, 2)
    val model1 = new DecisionTreeClassificationModel("dtc", TreeTests.root1, 3, 2)
    val model = new RandomForestClassificationModel("rfc", Array(model0, model1), 3, 2)
    model.setLeafCol("predictedLeafId")
      .setRawPredictionCol("")
      .setPredictionCol("")
      .setProbabilityCol("")

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
    val rf = new RandomForestClassifier().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[RandomForestClassificationModel, RandomForestClassifier](
      rf, spark) { (expected, actual) =>
      TreeTests.checkEqual(expected, actual)
    }
  }

  test("tree params") {
    val rdd = orderedLabeledPoints5_20
    val rf = new RandomForestClassifier()
      .setImpurity("entropy")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = TreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)
    model.setLeafCol("predictedLeafId")

    val transformed = model.transform(df)
    checkNominalOnDF(transformed, "prediction", model.numClasses)
    checkVectorSizeOnDF(transformed, "predictedLeafId", model.trees.length)
    checkVectorSizeOnDF(transformed, "rawPrediction", model.numClasses)
    checkVectorSizeOnDF(transformed, "probability", model.numClasses)

    model.trees.foreach (i => {
      assert(i.getMaxDepth === model.getMaxDepth)
      assert(i.getSeed === model.getSeed)
      assert(i.getImpurity === model.getImpurity)
    })
  }

  test("training with sample weights") {
    val df = binaryDataset
    val numClasses = 2
    // (numTrees, maxDepth, subsamplingRate, fractionInTol)
    val testParams = Seq(
      (1, 5, 1.0, 0.96),
      (1, 10, 1.0, 0.96),
      (1, 10, 0.95, 0.96)
    )

    for ((numTrees, maxDepth, subsamplingRate, tol) <- testParams) {
      val estimator = new RandomForestClassifier()
        .setNumTrees(numTrees)
        .setMaxDepth(maxDepth)
        .setSubsamplingRate(subsamplingRate)
        .setSeed(seed)
        .setMinWeightFractionPerNode(0.049)

      MLTestingUtils.testArbitrarilyScaledWeights[RandomForestClassificationModel,
        RandomForestClassifier](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, _ == _, tol))
      MLTestingUtils.testOutliersWithSmallWeights[RandomForestClassificationModel,
        RandomForestClassifier](df.as[LabeledPoint], estimator,
        numClasses, MLTestingUtils.modelPredictionEquals(df, _ == _, tol),
        outlierRatio = 2)
      MLTestingUtils.testOversamplingVsWeighting[RandomForestClassificationModel,
        RandomForestClassifier](df.as[LabeledPoint], estimator,
        MLTestingUtils.modelPredictionEquals(df, _ == _, tol), seed)
    }
  }

  test("summary for binary and multiclass") {
    val arr = new Array[LabeledPoint](300)
    for (i <- 0 until 300) {
      if (i < 100) {
        arr(i) = new LabeledPoint(0.0, Vectors.dense(2.0, 2.0))
      } else if (i < 200) {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(1.0, 2.0))
      } else {
        arr(i) = new LabeledPoint(2.0, Vectors.dense(0.0, 2.0))
      }
    }
    val rdd = sc.parallelize(arr)
    val multinomialDataset = spark.createDataFrame(rdd)

    val rf = new RandomForestClassifier()

    val brfModel = rf.fit(binaryDataset)
    assert(brfModel.summary.isInstanceOf[BinaryRandomForestClassificationTrainingSummary])
    assert(brfModel.summary.asBinary.isInstanceOf[BinaryRandomForestClassificationTrainingSummary])
    assert(brfModel.binarySummary.isInstanceOf[RandomForestClassificationTrainingSummary])
    assert(brfModel.summary.totalIterations === 0)
    assert(brfModel.binarySummary.totalIterations === 0)

    val mrfModel = rf.fit(multinomialDataset)
    assert(mrfModel.summary.isInstanceOf[RandomForestClassificationTrainingSummary])
    withClue("cannot get binary summary for multiclass model") {
      intercept[RuntimeException] {
        mrfModel.binarySummary
      }
    }
    withClue("cannot cast summary to binary summary multiclass model") {
      intercept[RuntimeException] {
        mrfModel.summary.asBinary
      }
    }
    assert(mrfModel.summary.totalIterations === 0)

    val brfSummary = brfModel.evaluate(binaryDataset)
    val mrfSummary = mrfModel.evaluate(multinomialDataset)
    assert(brfSummary.isInstanceOf[BinaryRandomForestClassificationSummary])
    assert(mrfSummary.isInstanceOf[RandomForestClassificationSummary])

    assert(brfSummary.accuracy === brfModel.summary.accuracy)
    assert(brfSummary.weightedPrecision === brfModel.summary.weightedPrecision)
    assert(brfSummary.weightedRecall === brfModel.summary.weightedRecall)
    assert(brfSummary.asBinary.areaUnderROC ~== brfModel.summary.asBinary.areaUnderROC relTol 1e-6)

    // verify instance weight works
    val rf2 = new RandomForestClassifier()
      .setWeightCol("weight")

    val binaryDatasetWithWeight =
      binaryDataset.select(col("label"), col("features"), lit(2.5).as("weight"))

    val multinomialDatasetWithWeight =
      multinomialDataset.select(col("label"), col("features"), lit(10.0).as("weight"))

    val brfModel2 = rf2.fit(binaryDatasetWithWeight)
    assert(brfModel2.summary.isInstanceOf[BinaryRandomForestClassificationTrainingSummary])
    assert(brfModel2.summary.asBinary.isInstanceOf[BinaryRandomForestClassificationTrainingSummary])
    assert(brfModel2.binarySummary.isInstanceOf[BinaryRandomForestClassificationTrainingSummary])

    val mrfModel2 = rf2.fit(multinomialDatasetWithWeight)
    assert(mrfModel2.summary.isInstanceOf[RandomForestClassificationTrainingSummary])
    withClue("cannot get binary summary for multiclass model") {
      intercept[RuntimeException] {
        mrfModel2.binarySummary
      }
    }
    withClue("cannot cast summary to binary summary multiclass model") {
      intercept[RuntimeException] {
        mrfModel2.summary.asBinary
      }
    }

    val brfSummary2 = brfModel2.evaluate(binaryDatasetWithWeight)
    val mrfSummary2 = mrfModel2.evaluate(multinomialDatasetWithWeight)
    assert(brfSummary2.isInstanceOf[BinaryRandomForestClassificationSummary])
    assert(mrfSummary2.isInstanceOf[RandomForestClassificationSummary])

    assert(brfSummary2.accuracy === brfModel2.summary.accuracy)
    assert(brfSummary2.weightedPrecision === brfModel2.summary.weightedPrecision)
    assert(brfSummary2.weightedRecall === brfModel2.summary.weightedRecall)
    assert(brfSummary2.asBinary.areaUnderROC ~==
      brfModel2.summary.asBinary.areaUnderROC relTol 1e-6)

    assert(brfSummary.accuracy ~== brfSummary2.accuracy relTol 1e-6)
    assert(brfSummary.weightedPrecision ~== brfSummary2.weightedPrecision relTol 1e-6)
    assert(brfSummary.weightedRecall ~== brfSummary2.weightedRecall relTol 1e-6)
    assert(brfSummary.asBinary.areaUnderROC ~== brfSummary2.asBinary.areaUnderROC relTol 1e-6)

    assert(brfModel.summary.asBinary.accuracy ~==
      brfModel2.summary.asBinary.accuracy relTol 1e-6)
    assert(brfModel.summary.asBinary.weightedPrecision ~==
      brfModel2.summary.asBinary.weightedPrecision relTol 1e-6)
    assert(brfModel.summary.asBinary.weightedRecall ~==
      brfModel2.summary.asBinary.weightedRecall relTol 1e-6)
    assert(brfModel.summary.asBinary.areaUnderROC ~==
      brfModel2.summary.asBinary.areaUnderROC relTol 1e-6)

    assert(mrfSummary.accuracy ~== mrfSummary2.accuracy relTol 1e-6)
    assert(mrfSummary.weightedPrecision ~== mrfSummary2.weightedPrecision relTol 1e-6)
    assert(mrfSummary.weightedRecall ~== mrfSummary2.weightedRecall relTol 1e-6)

    assert(mrfModel.summary.accuracy ~== mrfModel2.summary.accuracy relTol 1e-6)
    assert(mrfModel.summary.weightedPrecision ~== mrfModel2.summary.weightedPrecision relTol 1e-6)
    assert(mrfModel.summary.weightedRecall ~==mrfModel2.summary.weightedRecall relTol 1e-6)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
                        model: RandomForestClassificationModel,
                        model2: RandomForestClassificationModel): Unit = {
      TreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
      assert(model.numClasses === model2.numClasses)
    }

    val rf = new RandomForestClassifier().setNumTrees(2)
    val rdd = TreeTests.getTreeReadWriteData(sc)

    val allParamSettings = TreeTests.allParamSettings ++ Map("impurity" -> "entropy")

    val continuousData: DataFrame =
      TreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 2)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings,
      allParamSettings, checkModelData)
  }

  test("SPARK-33398: Load RandomForestClassificationModel prior to Spark 3.0") {
    val path = testFile("ml-models/rfc-2.4.7")
    val model = RandomForestClassificationModel.load(path)
    assert(model.numClasses === 2)
    assert(model.numFeatures === 692)
    assert(model.getNumTrees === 2)
    assert(model.totalNumNodes === 10)
    assert(model.trees.map(_.numNodes) === Array(3, 7))

    val metadata = spark.read.json(s"$path/metadata")
    val sparkVersionStr = metadata.select("sparkVersion").first().getString(0)
    assert(sparkVersionStr === "2.4.7")
  }
}
