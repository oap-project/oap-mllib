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

package org.apache.spark.ml.regression.spark321

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.classification.{LearningNode => LearningNodeDAL}
import com.intel.oap.mllib.regression.{RandomForestRegressorDALImpl, RandomForestRegressorShim}
import java.util.{Map => JavaMap}
import scala.jdk.CollectionConverters._

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.{DecisionTreeRegressionModel, RandomForestRegressionModel, Regressor}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.{DecisionTreeMetadata, RandomForest}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.sql.{Column, DataFrame, Dataset}
// scalastyle:off line.size.limit

/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 */
@Since("1.4.0")
class RandomForestRegressor @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends Regressor[Vector, RandomForestRegressor, RandomForestRegressionModel]
    with RandomForestRegressorParams with DefaultParamsWritable with RandomForestRegressorShim{

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("rfr"))

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }
  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeRegressorParams:

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
  override def train(dataset: Dataset[_]): RandomForestRegressionModel = instrumented { instr =>
    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      dataset.sparkSession.sparkContext)
    val model = if (Utils.isOAPEnabled() && isPlatformSupported) {
      trainRandomForestRegressorDAL(dataset, instr)
    } else {
      trainDiscreteImpl(dataset, instr)
    }
    model
  }


  private def trainRandomForestRegressorDAL(dataset: Dataset[_],
                                             instr: Instrumentation): RandomForestRegressionModel
  = {
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val spark = dataset.sparkSession
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val instances = extractInstances(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)
    strategy.bootstrap = $(bootstrap)

    instr.logDataset(instances)
    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, leafCol, impurity,
      numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, minWeightFractionPerNode, seed, subsamplingRate, cacheNodeIds,
      checkpointInterval, bootstrap)
    val metadata = DecisionTreeMetadata
      .buildMetadata(instances.retag(classOf[Instance]),
        strategy, getNumTrees, getFeatureSubsetStrategy)

    import spark.implicits._

    val sc = spark.sparkContext

    val executorNum = Utils.sparkExecutorNum(sc)
    val executorCores = Utils.sparkExecutorCores()
    val initStartTime = System.nanoTime()

    val rfDAL = new RandomForestRegressorDALImpl(uid,
      getNumTrees,
      metadata.numFeaturesPerNode,
      1,
      1,
      getMinWeightFractionPerNode,
      0.0,
      executorNum,
      executorCores,
      getBootstrap)

    val (predictionNumericTable, treesMap) = rfDAL.train(dataset, ${labelCol}, ${featuresCol})


    val numFeatures = metadata.numFeatures

    val trees = buildTrees(treesMap, numFeatures, metadata)
      .map(_.asInstanceOf[DecisionTreeRegressionModel])
    instr.logNumFeatures(numFeatures)
    new RandomForestRegressionModel(uid, trees, numFeatures)
  }

  private def buildTrees(treesMap : JavaMap[Integer,
    java.util.ArrayList[LearningNodeDAL]],
                         numFeatures : Int,
                         metadata: DecisionTreeMetadata): Array[DecisionTreeModel] = {
    treesMap.asScala.toMap.map { case (id: Integer, nodelist: java.util.ArrayList[LearningNodeDAL])
    => {
      val rootNode = TreeUtils.buildTreeDFS(nodelist, metadata)
      new DecisionTreeRegressionModel(uid,
        rootNode.toNode(),
        numFeatures)
    }
    }.toArray
  }

  private def trainDiscreteImpl(dataset: Dataset[_],
                                instr: Instrumentation): RandomForestRegressionModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val instances = extractInstances(dataset)
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)
    strategy.bootstrap = $(bootstrap)

    instr.logPipelineStage(this)
    instr.logDataset(instances)
    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, leafCol, impurity,
      numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, minWeightFractionPerNode, seed, subsamplingRate, cacheNodeIds,
      checkpointInterval, bootstrap)

    val trees = RandomForest
      .run(instances, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))
      .map(_.asInstanceOf[DecisionTreeRegressionModel])
    trees.foreach(copyValues(_))

    val numFeatures = trees.head.numFeatures
    instr.logNamedValue(Instrumentation.loggerTags.numFeatures, numFeatures)
    new RandomForestRegressionModel(uid, trees, numFeatures)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): RandomForestRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object RandomForestRegressor extends DefaultParamsReadable[RandomForestRegressor]{
  /** Accessor for supported impurity settings: variance */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = HasVarianceImpurity.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): RandomForestRegressor = super.load(path)

}

