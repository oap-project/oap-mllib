// scalastyle:off
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
// scalastyle:on

package org.apache.spark.ml.classification.spark321

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.classification.{RandomForestClassifierDALImpl, RandomForestClassifierShim, RandomForestResult, LearningNode => LearningNodeDAL}

import java.util.{ArrayList, Map => JavaMap}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import scala.jdk.CollectionConverters._
import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification.{BinaryClassificationSummary, BinaryRandomForestClassificationTrainingSummaryImpl, ClassificationSummary, DecisionTreeClassificationModel, ProbabilisticClassificationModel, ProbabilisticClassifier, RandomForestClassificationModel, RandomForestClassificationTrainingSummaryImpl, TrainingSummary}
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.{DecisionTreeModel, TreeClassifierParams, TreeEnsembleModel, _}
import org.apache.spark.ml.tree.impl.{DecisionTreeMetadata, RandomForest}
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.impurity.{GiniCalculator, ImpurityCalculator}
import org.apache.spark.mllib.tree.model.{ImpurityStats, RandomForestModel => OldRandomForestModel}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType

import scala.collection.JavaConversions.mapAsScalaMap
// scalastyle:off line.size.limit

/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> learning algorithm for
 * classification.
 * It supports both binary and multiclass labels, as well as both continuous and categorical
 * features.
 */
@Since("1.4.0")
class RandomForestClassifier @Since("1.4.0") (
                                               @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, RandomForestClassifier, RandomForestClassificationModel]
    with RandomForestClassifierParams with DefaultParamsWritable with RandomForestClassifierShim {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("rfc"))

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

  /** @group setParam */
  @Since("1.4.0")
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("3.0.0")
  def setMinWeightFractionPerNode(value: Double): this.type = set(minWeightFractionPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  @Since("1.4.0")
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  def setImpurity(value: String): this.type = set(impurity, value)

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  @Since("1.4.0")
  def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  @Since("3.0.0")
  def setBootstrap(value: Boolean): this.type = set(bootstrap, value)

  /** @group setParam */
  @Since("1.4.0")
  def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * By default the weightCol is not set, so all instances have weight 1.0.
   *
   * @group setParam
   */
  @Since("3.0.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override def train(dataset: Dataset[_]): RandomForestClassificationModel = instrumented { instr =>
    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      dataset.sparkSession.sparkContext)
    val model = if (Utils.isOAPEnabled() && isPlatformSupported) {
      trainRandomForestClassifierDAL(dataset, instr)
    } else {
      trainDiscreteImpl(dataset, instr)
    }
    model
  }


  private def trainRandomForestClassifierDAL(dataset: Dataset[_],
                       instr: Instrumentation): RandomForestClassificationModel = {
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val spark = dataset.sparkSession
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }
    val instances = extractInstances(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)
    strategy.bootstrap = $(bootstrap)

    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, probabilityCol,
      rawPredictionCol, leafCol, impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins,
      maxMemoryInMB, minInfoGain, minInstancesPerNode, minWeightFractionPerNode, seed,
      subsamplingRate, thresholds, cacheNodeIds, checkpointInterval, bootstrap)
    val metadata = DecisionTreeMetadata
      .buildMetadata(instances.retag(classOf[Instance]), strategy, getNumTrees, getFeatureSubsetStrategy)
    import spark.implicits._

    val sc = spark.sparkContext

    val executorNum = Utils.sparkExecutorNum(sc)
    val executorCores = Utils.sparkExecutorCores()

    val initStartTime = System.nanoTime()
    println(s"trainRandomForestClassifierDAL numClasses : " + numClasses)

    val rfDAL = new RandomForestClassifierDALImpl(uid,
      numClasses,
      getNumTrees,
      metadata.numFeaturesPerNode,
      1,
      1,
      getMinWeightFractionPerNode,
      0.0,
      executorNum,
      executorCores,
      getBootstrap)

    val (probabilitiesNumericTable, predictionNumericTable, treesMap) = rfDAL.train(dataset, ${labelCol}, ${featuresCol})


    val numFeatures = metadata.numFeatures

    val trees = buildTrees(treesMap, numFeatures, numClasses).map(_.asInstanceOf[DecisionTreeClassificationModel])
    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)
    createModel(dataset, trees, numFeatures, numClasses)
  }

  private def buildTrees(treesMap : JavaMap[Integer,
                         java.util.ArrayList[LearningNodeDAL]],
                         numFeatures : Int,
                         numClasses : Int): Array[DecisionTreeModel] = {
    treesMap.map { case (id: Integer, nodelist: java.util.ArrayList[LearningNodeDAL])
      => {
        val rootNode = TreeUtils.buildTreeDFS(nodelist)
        new DecisionTreeClassificationModel(uid,
                                            rootNode.toNode(),
                                            numFeatures,
                                            numClasses)
      }
    }.toArray
  }

  private def trainDiscreteImpl(dataset: Dataset[_],
                                instr: Instrumentation): RandomForestClassificationModel = {
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }
    val instances = extractInstances(dataset, numClasses)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)
    strategy.bootstrap = $(bootstrap)

    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, probabilityCol,
      rawPredictionCol, leafCol, impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins,
      maxMemoryInMB, minInfoGain, minInstancesPerNode, minWeightFractionPerNode, seed,
      subsamplingRate, thresholds, cacheNodeIds, checkpointInterval, bootstrap)

    val trees = RandomForest
      .run(instances, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeClassificationModel])
    trees.foreach(copyValues(_))

    val numFeatures = trees.head.numFeatures
    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)
    createModel(dataset, trees, numFeatures, numClasses)
  }

  private def createModel(
                           dataset: Dataset[_],
                           trees: Array[DecisionTreeClassificationModel],
                           numFeatures: Int,
                           numClasses: Int): RandomForestClassificationModel = {
    val model = copyValues(new RandomForestClassificationModel(uid, trees, numFeatures, numClasses))
    val weightColName = if (!isDefined(weightCol)) "weightCol" else $(weightCol)

    val (summaryModel, probabilityColName, predictionColName) = model.findSummaryModel()
    val rfSummary = if (numClasses <= 2) {
      new BinaryRandomForestClassificationTrainingSummaryImpl(
        summaryModel.transform(dataset),
        probabilityColName,
        predictionColName,
        $(labelCol),
        weightColName,
        Array(0.0))
    } else {
      new RandomForestClassificationTrainingSummaryImpl(
        summaryModel.transform(dataset),
        predictionColName,
        $(labelCol),
        weightColName,
        Array(0.0))
    }
    model.setSummary(Some(rfSummary))
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): RandomForestClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object RandomForestClassifier extends DefaultParamsReadable[RandomForestClassifier] {
  /** Accessor for supported impurity settings: entropy, gini */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeClassifierParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): RandomForestClassifier = super.load(path)
}

