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

package org.apache.spark.ml.classification.spark320

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.classification.{NaiveBayesDALImpl, NaiveBayesShim}

import org.apache.spark.annotation.Since
import org.apache.spark.ml.classification._
import org.apache.spark.ml.classification.{NaiveBayes => SparkNaiveBayes}
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

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
class NaiveBayes @Since("1.5.0")(@Since("1.5.0") override val uid: String)
    extends SparkNaiveBayes
    with NaiveBayesShim {

  import SparkNaiveBayes._
  import NaiveBayes._

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("nb"))

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }
  override def train(dataset: Dataset[_]): NaiveBayesModel = {
    trainWithLabelCheck(dataset, positiveLabel = true)
  }

  /**
   * ml assumes input labels in range [0, numClasses). But this implementation
   * is also called by mllib NaiveBayes which allows other kinds of input labels
   * such as {-1, +1}. `positiveLabel` is used to determine whether the label
   * should be checked and it should be removed when we remove mllib NaiveBayes.
   */
  private[spark] def trainWithLabelCheck(
      dataset: Dataset[_],
      positiveLabel: Boolean): NaiveBayesModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(
      this,
      labelCol,
      featuresCol,
      weightCol,
      predictionCol,
      rawPredictionCol,
      probabilityCol,
      modelType,
      smoothing,
      thresholds)

    if (positiveLabel && isDefined(thresholds)) {
      val numClasses = getNumClasses(dataset)
      instr.logNumClasses(numClasses)
      require(
        $(thresholds).length == numClasses,
        this.getClass.getSimpleName +
          ".train() called with non-matching numClasses and thresholds.length." +
          s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    $(modelType) match {
      case Multinomial =>
        val sc = dataset.sparkSession.sparkContext
        val model = if (Utils.isOAPEnabled()) {
          val isPlatformSupported =
            Utils.checkClusterPlatformCompatibility(dataset.sparkSession.sparkContext)
          val handleWeight = (isDefined(weightCol) && $(weightCol).nonEmpty)
          val handleSmoothing = ($(smoothing) != 1.0)
          if (isPlatformSupported && !handleWeight && !handleSmoothing) {
            trainNaiveBayesDAL(dataset, instr)
          } else {
            trainDiscreteImpl(dataset, instr)
          }
        } else {
          trainDiscreteImpl(dataset, instr)
        }
        model
      case Bernoulli | Complement =>
        trainDiscreteImpl(dataset, instr)
      case Gaussian =>
        trainGaussianImpl(dataset, instr)
      case _ =>
        // This should never happen.
        throw new IllegalArgumentException(s"Invalid modelType: ${$(modelType)}.")
    }
  }

  private def trainNaiveBayesDAL(dataset: Dataset[_], instr: Instrumentation): NaiveBayesModel = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val sc = spark.sparkContext

    val executor_num = Utils.sparkExecutorNum(sc)
    val executor_cores = Utils.sparkExecutorCores()

    logInfo(s"NaiveBayesDAL fit using $executor_num Executors")

    // DAL only support [0..numClasses) as labels, should map original labels using StringIndexer
    // Todo: optimize getting num of classes
    // A temp spark config to specify numClasses, may be removed in the future
    val confClasses = sc.conf.getInt("spark.oap.mllib.classification.classes", -1)

    // numClasses should be explicitly included in the parquet metadata
    // This can be done by applying StringIndexer to the label column
    val numClasses = confClasses match {
      case -1 => getNumClasses(dataset)
      case _ => confClasses
    }

    instr.logNumClasses(numClasses)

    val labeledPointsDS = dataset
      .select(col(getLabelCol), DatasetUtils.columnToVector(dataset, getFeaturesCol))

    val dalModel = new NaiveBayesDALImpl(uid, numClasses, executor_num, executor_cores)
      .train(labeledPointsDS, $ { labelCol }, $ { featuresCol })

    val model = copyValues(
      new NaiveBayesModel(dalModel.uid, dalModel.pi, dalModel.theta, dalModel.sigma))

    // Set labels to be compatible with old mllib model
    val labels = (0 until numClasses).map(_.toDouble).toArray
    model.setOldLabels(labels)

    model
  }

  private def trainDiscreteImpl(dataset: Dataset[_], instr: Instrumentation): NaiveBayesModel = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val validateUDF = $(modelType) match {
      case Multinomial | Complement =>
        udf { vector: Vector =>
          requireNonnegativeValues(vector); vector
        }
      case Bernoulli =>
        udf { vector: Vector =>
          requireZeroOneBernoulliValues(vector); vector
        }
    }

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      checkNonNegativeWeight(col($(weightCol)).cast(DoubleType))
    } else {
      lit(1.0)
    }

    // Aggregates term frequencies per label.
    val aggregated = dataset
      .groupBy(col($(labelCol)))
      .agg(
        sum(w).as("weightSum"),
        Summarizer
          .metrics("sum", "count")
          .summary(validateUDF(col($(featuresCol))), w)
          .as("summary"))
      .select($(labelCol), "weightSum", "summary.sum", "summary.count")
      .as[(Double, Double, Vector, Long)]
      .collect()
      .sortBy(_._1)

    val numFeatures = aggregated.head._3.size
    instr.logNumFeatures(numFeatures)
    val numSamples = aggregated.map(_._4).sum
    instr.logNumExamples(numSamples)
    val numLabels = aggregated.length
    instr.logNumClasses(numLabels)
    val numDocuments = aggregated.map(_._2).sum
    instr.logSumOfWeights(numDocuments)

    val labelArray = new Array[Double](numLabels)
    val piArray = new Array[Double](numLabels)
    val thetaArray = new Array[Double](numLabels * numFeatures)

    val aggIter = $(modelType) match {
      case Multinomial | Bernoulli => aggregated.iterator
      case Complement =>
        val featureSum = Vectors.zeros(numFeatures)
        aggregated.foreach {
          case (_, _, sumTermFreqs, _) =>
            BLAS.axpy(1.0, sumTermFreqs, featureSum)
        }
        aggregated.iterator.map {
          case (label, n, sumTermFreqs, count) =>
            val comp = featureSum.copy
            BLAS.axpy(-1.0, sumTermFreqs, comp)
            (label, n, comp, count)
        }
    }

    val lambda = $(smoothing)
    val piLogDenom = math.log(numDocuments + numLabels * lambda)
    var i = 0
    aggIter.foreach {
      case (label, n, sumTermFreqs, _) =>
        labelArray(i) = label
        piArray(i) = math.log(n + lambda) - piLogDenom
        val thetaLogDenom = $(modelType) match {
          case Multinomial | Complement =>
            math.log(sumTermFreqs.toArray.sum + numFeatures * lambda)
          case Bernoulli => math.log(n + 2.0 * lambda)
        }
        var j = 0
        val offset = i * numFeatures
        while (j < numFeatures) {
          thetaArray(offset + j) = math.log(sumTermFreqs(j) + lambda) - thetaLogDenom
          j += 1
        }
        i += 1
    }

    val pi = Vectors.dense(piArray)
    $(modelType) match {
      case Multinomial | Bernoulli =>
        val theta = new DenseMatrix(numLabels, numFeatures, thetaArray, true)
        new NaiveBayesModel(uid, pi.compressed, theta.compressed, Matrices.zeros(0, 0))
          .setOldLabels(labelArray)
      case Complement =>
        // Since the CNB compute the coefficient in a complement way.
        val theta = new DenseMatrix(numLabels, numFeatures, thetaArray.map(v => -v), true)
        new NaiveBayesModel(uid, pi.compressed, theta.compressed, Matrices.zeros(0, 0))
    }
  }

  private def trainGaussianImpl(dataset: Dataset[_], instr: Instrumentation): NaiveBayesModel = {
    val spark = dataset.sparkSession
    import spark.implicits._

    val w = if (isDefined(weightCol) && $(weightCol).nonEmpty) {
      checkNonNegativeWeight(col($(weightCol)).cast(DoubleType))
    } else {
      lit(1.0)
    }

    // Aggregates mean vector and square-sum vector per label.
    val aggregated = dataset
      .groupBy(col($(labelCol)))
      .agg(
        sum(w).as("weightSum"),
        Summarizer
          .metrics("mean", "normL2")
          .summary(col($(featuresCol)), w)
          .as("summary"))
      .select($(labelCol), "weightSum", "summary.mean", "summary.normL2")
      .as[(Double, Double, Vector, Vector)]
      .map {
        case (label, weightSum, mean, normL2) =>
          (label, weightSum, mean, Vectors.dense(normL2.toArray.map(v => v * v)))
      }
      .collect()
      .sortBy(_._1)

    val numFeatures = aggregated.head._3.size
    instr.logNumFeatures(numFeatures)

    val numLabels = aggregated.length
    instr.logNumClasses(numLabels)

    val numInstances = aggregated.map(_._2).sum
    instr.logSumOfWeights(numInstances)

    // If the ratio of data variance between dimensions is too small, it
    // will cause numerical errors. To address this, we artificially
    // boost the variance by epsilon, a small fraction of the standard
    // deviation of the largest dimension.
    // Refer to scikit-learn's implementation
    // [https://github.com/scikit-learn/scikit-learn/blob/0.21.X/sklearn/naive_bayes.py#L348]
    // and discussion [https://github.com/scikit-learn/scikit-learn/pull/5349] for detail.
    val epsilon = Iterator
      .range(0, numFeatures)
      .map { j =>
        var globalSum = 0.0
        var globalSqrSum = 0.0
        aggregated.foreach {
          case (_, weightSum, mean, squareSum) =>
            globalSum += mean(j) * weightSum
            globalSqrSum += squareSum(j)
        }
        globalSqrSum / numInstances -
          globalSum * globalSum / numInstances / numInstances
      }
      .max * 1e-9

    val piArray = new Array[Double](numLabels)

    // thetaArray in Gaussian NB store the means of features per label
    val thetaArray = new Array[Double](numLabels * numFeatures)

    // thetaArray in Gaussian NB store the variances of features per label
    val sigmaArray = new Array[Double](numLabels * numFeatures)

    var i = 0
    val logNumInstances = math.log(numInstances)
    aggregated.foreach {
      case (_, weightSum, mean, squareSum) =>
        piArray(i) = math.log(weightSum) - logNumInstances
        var j = 0
        val offset = i * numFeatures
        while (j < numFeatures) {
          val m = mean(j)
          thetaArray(offset + j) = m
          sigmaArray(offset + j) = epsilon + squareSum(j) / weightSum - m * m
          j += 1
        }
        i += 1
    }

    val pi = Vectors.dense(piArray)
    val theta = new DenseMatrix(numLabels, numFeatures, thetaArray, true)
    val sigma = new DenseMatrix(numLabels, numFeatures, sigmaArray, true)
    new NaiveBayesModel(uid, pi.compressed, theta.compressed, sigma.compressed)
  }
}
