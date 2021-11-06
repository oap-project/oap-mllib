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

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.classification.NaiveBayesShim
import org.apache.hadoop.fs.Path
import org.json4s.DefaultFormats
import org.apache.spark.annotation.Since
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.util.VersionUtils

// scalastyle:off line.size.limit
/**
 * Naive Bayes Classifiers.
 * It supports Multinomial NB
 * (see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html">
 * here</a>)
 * which can handle finitely supported discrete data. For example, by converting documents into
 * TF-IDF vectors, it can be used for document classification. By making every vector a
 * binary (0/1) data, it can also be used as Bernoulli NB
 * (see <a href="http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html">
 * here</a>).
 * The input feature values for Multinomial NB and Bernoulli NB must be nonnegative.
 * Since 3.0.0, it supports Complement NB which is an adaptation of the Multinomial NB. Specifically,
 * Complement NB uses statistics from the complement of each class to compute the model's coefficients
 * The inventors of Complement NB show empirically that the parameter estimates for CNB are more stable
 * than those for Multinomial NB. Like Multinomial NB, the input feature values for Complement NB must
 * be nonnegative.
 * Since 3.0.0, it also supports Gaussian NB
 * (see <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes">
 * here</a>)
 * which can handle continuous data.
 */
// scalastyle:on line.size.limit
@Since("1.5.0")
class NaiveBayes @Since("1.5.0") (
    @Since("1.5.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, NaiveBayes, NaiveBayesModel]
  with NaiveBayesParams with DefaultParamsWritable {

  import NaiveBayes._

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("nb"))

  /**
   * Set the smoothing parameter.
   * Default is 1.0.
   * @group setParam
   */
  @Since("1.5.0")
  def setSmoothing(value: Double): this.type = set(smoothing, value)
  setDefault(smoothing -> 1.0)

  /**
   * Set the model type using a string (case-sensitive).
   * Supported options: "multinomial", "complement", "bernoulli", and "gaussian".
   * Default is "multinomial"
   * @group setParam
   */
  @Since("1.5.0")
  def setModelType(value: String): this.type = set(modelType, value)
  setDefault(modelType -> Multinomial)

  /**
   * Sets the value of param [[weightCol]].
   * If this is not set or empty, we treat all instance weights as 1.0.
   * Default is not set, so all instances have weight one.
   *
   * @group setParam
   */
  @Since("2.1.0")
  def setWeightCol(value: String): this.type = set(weightCol, value)

  override protected def train(dataset: Dataset[_]): NaiveBayesModel = {
    val naiveBayes = NaiveBayesShim.create(uid)
      .setModelType($(modelType))
      .setSmoothing($(smoothing))
    if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      naiveBayes.setWeightCol($(weightCol))
    }
    naiveBayes.train(dataset)
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): NaiveBayes = defaultCopy(extra)
}

@Since("1.6.0")
object NaiveBayes extends DefaultParamsReadable[NaiveBayes] {
  /** String name for multinomial model type. */
  private[classification] val Multinomial: String = "multinomial"

  /** String name for Bernoulli model type. */
  private[classification] val Bernoulli: String = "bernoulli"

  /** String name for Gaussian model type. */
  private[classification] val Gaussian: String = "gaussian"

  /** String name for Complement model type. */
  private[classification] val Complement: String = "complement"

  /* Set of modelTypes that NaiveBayes supports */
  private[classification] val supportedModelTypes =
    Set(Multinomial, Bernoulli, Gaussian, Complement)

  private[ml] def requireNonnegativeValues(v: Vector): Unit = {
    require(v.nonZeroIterator.forall(_._2 > 0.0),
      s"Naive Bayes requires nonnegative feature values but found $v.")
  }

  private[ml] def requireZeroOneBernoulliValues(v: Vector): Unit = {
    require(v.nonZeroIterator.forall(_._2 == 1.0),
      s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
  }

  @Since("1.6.0")
  override def load(path: String): NaiveBayes = super.load(path)
}
