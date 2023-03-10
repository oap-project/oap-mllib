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

package org.apache.spark.ml.regression.spark313

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{
  CachedDiffFunction,
  DiffFunction,
  FirstOrderMinimizer,
  LBFGS => BreezeLBFGS,
  LBFGSB => BreezeLBFGSB,
  OWLQN => BreezeOWLQN
}
import breeze.stats.distributions.StudentsT
import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.regression.{LinearRegressionDALImpl, LinearRegressionShim}
import org.apache.hadoop.fs.Path
import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{PipelineStage, PredictorParams}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.BLAS._
import org.apache.spark.ml.optim.WeightedLeastSquares
import org.apache.spark.ml.optim.aggregator._
import org.apache.spark.ml.optim.loss.{L2Regularization, RDDLossFunction}
import org.apache.spark.ml.param.{DoubleParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.{LinearRegression => SparkLinearRegression, _}
import org.apache.spark.ml.stat._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.mllib.regression.{LinearRegressionModel => OldLinearRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DataType, DoubleType, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.VersionUtils.majorMinorVersion

/**
 * Params for linear regression.
 */
private[regression] trait LinearRegressionParams extends PredictorParams
    with HasRegParam with HasElasticNetParam with HasMaxIter with HasTol
    with HasFitIntercept with HasStandardization with HasWeightCol with HasSolver
    with HasAggregationDepth with HasLoss with HasMaxBlockSizeInMB {

  import LinearRegression._

  /**
   * The solver algorithm for optimization.
   * Supported options: "l-bfgs", "normal" and "auto".
   * Default: "auto"
   *
   * @group param
   */
  @Since("1.6.0")
  final override val solver: Param[String] = new Param[String](this, "solver",
    "The solver algorithm for optimization. Supported options: " +
      s"${supportedSolvers.mkString(", ")}. (Default auto)",
    ParamValidators.inArray[String](supportedSolvers))

  /**
   * The loss function to be optimized.
   * Supported options: "squaredError" and "huber".
   * Default: "squaredError"
   *
   * @group param
   */
  @Since("2.3.0")
  final override val loss: Param[String] = new Param[String](this, "loss", "The loss function to" +
    s" be optimized. Supported options: ${supportedLosses.mkString(", ")}. (Default squaredError)",
    ParamValidators.inArray[String](supportedLosses))

  /**
   * The shape parameter to control the amount of robustness. Must be &gt; 1.0.
   * At larger values of epsilon, the huber criterion becomes more similar to least squares
   * regression; for small values of epsilon, the criterion is more similar to L1 regression.
   * Default is 1.35 to get as much robustness as possible while retaining
   * 95% statistical efficiency for normally distributed data. It matches sklearn
   * HuberRegressor and is "M" from <a href="http://statweb.stanford.edu/~owen/reports/hhu.pdf">
   * A robust hybrid of lasso and ridge regression</a>.
   * Only valid when "loss" is "huber".
   *
   * @group expertParam
   */
  @Since("2.3.0")
  final val epsilon = new DoubleParam(this, "epsilon", "The shape parameter to control the " +
    "amount of robustness. Must be > 1.0.", ParamValidators.gt(1.0))

  /** @group getExpertParam */
  @Since("2.3.0")
  def getEpsilon: Double = $(epsilon)

  setDefault(regParam -> 0.0, fitIntercept -> true, standardization -> true,
    elasticNetParam -> 0.0, maxIter -> 100, tol -> 1E-6, solver -> Auto,
    aggregationDepth -> 2, loss -> SquaredError, epsilon -> 1.35, maxBlockSizeInMB -> 0.0)

  override protected def validateAndTransformSchema(
      schema: StructType,
      fitting: Boolean,
      featuresDataType: DataType): StructType = {
    if (fitting) {
      if ($(loss) == Huber) {
        require($(solver)!= Normal, "LinearRegression with huber loss doesn't support " +
          "normal solver, please change solver to auto or l-bfgs.")
        require($(elasticNetParam) == 0.0, "LinearRegression with huber loss only supports " +
          s"L2 regularization, but got elasticNetParam = $getElasticNetParam.")
      }
    }
    super.validateAndTransformSchema(schema, fitting, featuresDataType)
  }
}

/**
 * Linear regression.
 *
 * The learning objective is to minimize the specified loss function, with regularization.
 * This supports two kinds of loss:
 *  - squaredError (a.k.a squared loss)
 *  - huber (a hybrid of squared error for relatively small errors and absolute error for
 *  relatively large ones, and we estimate the scale parameter from training data)
 *
 * This supports multiple types of regularization:
 *  - none (a.k.a. ordinary least squares)
 *  - L2 (ridge regression)
 *  - L1 (Lasso)
 *  - L2 + L1 (elastic net)
 *
 * The squared error objective function is:
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   \min_{w}\frac{1}{2n}{\sum_{i=1}^n(X_{i}w - y_{i})^{2} +
 *   \lambda\left[\frac{1-\alpha}{2}{||w||_{2}}^{2} + \alpha{||w||_{1}}\right]}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * The huber objective function is:
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   \min_{w, \sigma}\frac{1}{2n}{\sum_{i=1}^n\left(\sigma +
 *   H_m\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \frac{1}{2}\lambda {||w||_2}^2}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * where
 *
 * <blockquote>
 *   $$
 *   \begin{align}
 *   H_m(z) = \begin{cases}
 *            z^2, & \text {if } |z| &lt; \epsilon, \\
 *            2\epsilon|z| - \epsilon^2, & \text{otherwise}
 *            \end{cases}
 *   \end{align}
 *   $$
 * </blockquote>
 *
 * Since 3.1.0, it supports stacking instances into blocks and using GEMV for
 * better performance.
 * The block size will be 1.0 MB, if param maxBlockSizeInMB is set 0.0 by default.
 *
 * Note: Fitting with huber loss only supports none and L2 regularization.
 */
@Since("1.3.0")
class LinearRegression @Since("1.3") (@Since("1.3.0") override val uid: String)
  extends SparkLinearRegression with LinearRegressionShim {

  import LinearRegression._

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("linReg"))

//  /**
//   * Set the regularization parameter.
//   * Default is 0.0.
//   *
//   * @group setParam
//   */
//  @Since("1.3.0")
//  def setRegParam(value: Double): this.type = set(regParam, value)
//
//  /**
//   * Set if we should fit the intercept.
//   * Default is true.
//   *
//   * @group setParam
//   */
//  @Since("1.5.0")
//  def setFitIntercept(value: Boolean): this.type = set(fitIntercept, value)
//
//  /**
//   * Whether to standardize the training features before fitting the model.
//   * The coefficients of models will be always returned on the original scale,
//   * so it will be transparent for users.
//   * Default is true.
//   *
//   * @note With/without standardization, the models should be always converged
//   * to the same solution when no regularization is applied. In R's GLMNET package,
//   * the default behavior is true as well.
//   *
//   * @group setParam
//   */
//  @Since("1.5.0")
//  def setStandardization(value: Boolean): this.type = set(standardization, value)
//
//  /**
//   * Set the ElasticNet mixing parameter.
//   * For alpha = 0, the penalty is an L2 penalty.
//   * For alpha = 1, it is an L1 penalty.
//   * For alpha in (0,1), the penalty is a combination of L1 and L2.
//   * Default is 0.0 which is an L2 penalty.
//   *
//   * Note: Fitting with huber loss only supports None and L2 regularization,
//   * so throws exception if this param is non-zero value.
//   *
//   * @group setParam
//   */
//  @Since("1.4.0")
//  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
//
//  /**
//   * Set the maximum number of iterations.
//   * Default is 100.
//   *
//   * @group setParam
//   */
//  @Since("1.3.0")
//  def setMaxIter(value: Int): this.type = set(maxIter, value)
//
//  /**
//   * Set the convergence tolerance of iterations.
//   * Smaller value will lead to higher accuracy with the cost of more iterations.
//   * Default is 1E-6.
//   *
//   * @group setParam
//   */
//  @Since("1.4.0")
//  def setTol(value: Double): this.type = set(tol, value)
//
//  /**
//   * Whether to over-/under-sample training instances according to the given weights in weightCol.
//   * If not set or empty, all instances are treated equally (weight 1.0).
//   * Default is not set, so all instances have weight one.
//   *
//   * @group setParam
//   */
//  @Since("1.6.0")
//  def setWeightCol(value: String): this.type = set(weightCol, value)
//
//  /**
//   * Set the solver algorithm used for optimization.
//   * In case of linear regression, this can be "l-bfgs", "normal" and "auto".
//   *  - "l-bfgs" denotes Limited-memory BFGS which is a limited-memory quasi-Newton
//   *    optimization method.
//   *  - "normal" denotes using Normal Equation as an analytical solution to the linear regression
//   *    problem.  This solver is limited to `LinearRegression.MAX_FEATURES_FOR_NORMAL_SOLVER`.
//   *  - "auto" (default) means that the solver algorithm is selected automatically.
//   *    The Normal Equations solver will be used when possible, but this will automatically fall
//   *    back to iterative optimization methods when needed.
//   *
//   * Note: Fitting with huber loss doesn't support normal solver,
//   * so throws exception if this param was set with "normal".
//   * @group setParam
//   */
//  @Since("1.6.0")
//  def setSolver(value: String): this.type = set(solver, value)
//
//  /**
//   * Suggested depth for treeAggregate (greater than or equal to 2).
//   * If the dimensions of features or the number of partitions are large,
//   * this param could be adjusted to a larger size.
//   * Default is 2.
//   *
//   * @group expertSetParam
//   */
//  @Since("2.1.0")
//  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)
//
//  /**
//   * Sets the value of param [[loss]].
//   * Default is "squaredError".
//   *
//   * @group setParam
//   */
//  @Since("2.3.0")
//  def setLoss(value: String): this.type = set(loss, value)
//
//  /**
//   * Sets the value of param [[epsilon]].
//   * Default is 1.35.
//   *
//   * @group setExpertParam
//   */
//  @Since("2.3.0")
//  def setEpsilon(value: Double): this.type = set(epsilon, value)
//
//  /**
//   * Sets the value of param [[maxBlockSizeInMB]].
//   * Default is 0.0, then 1.0 MB will be chosen.
//   *
//   * @group expertSetParam
//   */
//  @Since("3.1.0")
//  def setMaxBlockSizeInMB(value: Double): this.type = set(maxBlockSizeInMB, value)

  override def fit(
    dataset: Dataset[_]): LinearRegressionModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, solver, tol,
      elasticNetParam, fitIntercept, maxIter, regParam, standardization, aggregationDepth, loss,
      epsilon, maxBlockSizeInMB)

      println("right fit")

    val yMean = 0.0


    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    instr.logNumFeatures(numFeatures)

    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      dataset.sparkSession.sparkContext)
    val useLRDAL = Utils.isOAPEnabled() && isPlatformSupported

    val model = if (useLRDAL) {
      trainWithDAL(dataset, numFeatures, yMean)
    } else {
      // todo
      trainWithDAL(dataset, numFeatures, yMean)
    }

    model
  }

  //private def trainWithDAL(instances: RDD[(Vector, Double)]
  private def trainWithDAL(dataset: Dataset[_],
      numFeatures: Int,
      yMean: Double): LinearRegressionModel = instrumented { instr =>

    val sc = instances.sparkContext

    val executor_num = Utils.sparkExecutorNum(sc)
    val executor_cores = Utils.sparkExecutorCores()


    val coefficients = Vectors.sparse(numFeatures, Seq.empty)
    val intercept = yMean


    //prepare data
    val inputData = instances.map {
      case (point: Vector, weight: Double) => point
    }

    val lrDAL = new LRDALImpl(executor_num, executor_cores)

    //train
    val lrmodel = lrDAL.train(inputdata)

    val model = copyValues(new LinearRegressionModel(uid, model.coefficients, model.intercept).setParent(this))
    val trainingSummary = new LinearRegressionTrainingSummary(
      summaryModel.transform(dataset), predictionColName, $(labelCol), $(featuresCol),
      summaryModel, model.diagInvAtWA.toArray, model.objectiveHistory)

    model.setSummary(Some(trainingSummary))

  }
  /** todo
  private def trainWithML(dataset: Dataset[_],
      numFeatures: Int,
      yMean: Double): LinearRegressionModel = instrumented {
        trainWithDAL(dataset, numFeatures, yMean)
  }
  */

  override def train(
    dataset: Dataset[_]): LinearRegressionModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, weightCol, predictionCol, solver, tol,
      elasticNetParam, fitIntercept, maxIter, regParam, standardization, aggregationDepth, loss,
      epsilon, maxBlockSizeInMB)

    if (dataset.storageLevel != StorageLevel.NONE) {
      instr.logWarning(s"Input instances will be standardized, blockified to blocks, and " +
        s"then cached during training. Be careful of double caching!")
    }

    // Extract the number of features before deciding optimization solver.
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    instr.logNumFeatures(numFeatures)

    if ($(loss) == SquaredError && (($(solver) == Auto &&
      numFeatures <= WeightedLeastSquares.MAX_NUM_FEATURES) || $(solver) == Normal)) {
      return trainWithNormal(dataset, instr)
    }

    val instances = extractInstances(dataset)
      .setName("training instances")

    val (summarizer, labelSummarizer) = Summarizer
      .getRegressionSummarizers(instances, $(aggregationDepth), Seq("mean", "std", "count"))

    val yMean = labelSummarizer.mean(0)
    val rawYStd = labelSummarizer.std(0)

    instr.logNumExamples(labelSummarizer.count)
    instr.logNamedValue(Instrumentation.loggerTags.meanOfLabels, yMean)
    instr.logNamedValue(Instrumentation.loggerTags.varianceOfLabels, rawYStd)
    instr.logSumOfWeights(summarizer.weightSum)

    var actualBlockSizeInMB = $(maxBlockSizeInMB)
    if (actualBlockSizeInMB == 0) {
      actualBlockSizeInMB = InstanceBlock.DefaultBlockSizeInMB
      require(actualBlockSizeInMB > 0, "inferred actual BlockSizeInMB must > 0")
      instr.logNamedValue("actualBlockSizeInMB", actualBlockSizeInMB.toString)
    }

    if (rawYStd == 0.0) {
      if ($(fitIntercept) || yMean == 0.0) {
        return trainWithConstantLabel(dataset, instr, numFeatures, yMean)
      } else {
        require($(regParam) == 0.0, "The standard deviation of the label is zero. " +
          "Model cannot be regularized.")
        instr.logWarning(s"The standard deviation of the label is zero. " +
          "Consider setting fitIntercept=true.")
      }
    }

    // if y is constant (rawYStd is zero), then y cannot be scaled. In this case
    // setting yStd=abs(yMean) ensures that y is not scaled anymore in l-bfgs algorithm.
    val yStd = if (rawYStd > 0) rawYStd else math.abs(yMean)
    val featuresMean = summarizer.mean.toArray
    val featuresStd = summarizer.std.toArray

    if (!$(fitIntercept) && (0 until numFeatures).exists { i =>
      featuresStd(i) == 0.0 && featuresMean(i) != 0.0 }) {
      instr.logWarning("Fitting LinearRegressionModel without intercept on dataset with " +
        "constant nonzero column, Spark MLlib outputs zero coefficients for constant nonzero " +
        "columns. This behavior is the same as R glmnet but different from LIBSVM.")
    }

    // Since we implicitly do the feature scaling when we compute the cost function
    // to improve the convergence, the effective regParam will be changed.
    val effectiveRegParam = $(loss) match {
      case SquaredError => $(regParam) / yStd
      case Huber => $(regParam)
    }
    val effectiveL1RegParam = $(elasticNetParam) * effectiveRegParam
    val effectiveL2RegParam = (1.0 - $(elasticNetParam)) * effectiveRegParam

    val getFeaturesStd = (j: Int) => if (j >= 0 && j < numFeatures) featuresStd(j) else 0.0
    val regularization = if (effectiveL2RegParam != 0.0) {
      val shouldApply = (idx: Int) => idx >= 0 && idx < numFeatures
      Some(new L2Regularization(effectiveL2RegParam, shouldApply,
        if ($(standardization)) None else Some(getFeaturesStd)))
    } else None

    val optimizer = createOptimizer(effectiveRegParam, effectiveL1RegParam,
      numFeatures, featuresStd)

    val initialValues = $(loss) match {
      case SquaredError =>
        Vectors.zeros(numFeatures)
      case Huber =>
        val dim = if ($(fitIntercept)) numFeatures + 2 else numFeatures + 1
        Vectors.dense(Array.fill(dim)(1.0))
    }

    val (parameters, objectiveHistory) =
      trainImpl(instances, actualBlockSizeInMB, yMean, yStd,
        featuresMean, featuresStd, initialValues, regularization, optimizer)

    if (parameters == null) {
      val msg = s"${optimizer.getClass.getName} failed."
      instr.logError(msg)
      throw new SparkException(msg)
    }

    val model = createModel(parameters, yMean, yStd, featuresMean, featuresStd)
    // Handle possible missing or invalid prediction columns
    val (summaryModel, predictionColName) = model.findSummaryModelAndPredictionCol()

    val trainingSummary = new LinearRegressionTrainingSummary(
      summaryModel.transform(dataset), predictionColName, $(labelCol), $(featuresCol),
      model, Array(0.0), objectiveHistory)
    model.setSummary(Some(trainingSummary))
  }

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }

  private def trainWithNormal(
      dataset: Dataset[_],
      instr: Instrumentation): LinearRegressionModel = {
    // oneDAL only support simple linear regression and ridge regression
    val paramSupported = ($(regParam) == 0) || ($(regParam) != 0 && $(elasticNetParam) == 0)
    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      dataset.sparkSession.sparkContext)
    if (paramSupported && Utils.isOAPEnabled && isPlatformSupported) {
      val executor_num = Utils.sparkExecutorNum(dataset.sparkSession.sparkContext)
      val executor_cores = Utils.sparkExecutorCores()
      logInfo(s"LinearRegressionDAL fit using $executor_num Executors")

      val optimizer = new LinearRegressionDALImpl($(fitIntercept), $(regParam),
        elasticNetParam = $(elasticNetParam), $(standardization), true,
        executor_num, executor_cores)

      // Return same model as WeightedLeastSquaresModel
      val model = optimizer.train(dataset, $(labelCol), $(featuresCol))

      val lrModel = copyValues(
        new LinearRegressionModel(uid, model.coefficients, model.intercept))

      val (summaryModel, predictionColName) = lrModel.findSummaryModelAndPredictionCol()

      val trainingSummary = new LinearRegressionTrainingSummary(
        summaryModel.transform(dataset),
        predictionColName,
        $(labelCol),
        $(featuresCol),
        summaryModel,
        model.diagInvAtWA.toArray,
        model.objectiveHistory)

      return lrModel.setSummary(Some(trainingSummary))
    } else {
      // For low dimensional data, WeightedLeastSquares is more efficient since the
      // training algorithm only requires one pass through the data. (SPARK-10668)

      val optimizer = new WeightedLeastSquares($(fitIntercept), $(regParam),
        elasticNetParam = $(elasticNetParam), $(standardization), true,
        solverType = WeightedLeastSquares.Auto, maxIter = $(maxIter), tol = $(tol))
      val instances = extractInstances(dataset)
        .setName("training instances")
      val model = optimizer.fit(instances, instr = OptionalInstrumentation.create(instr))
      // When it is trained by WeightedLeastSquares, training summary does not
      // attach returned model.
      val lrModel = copyValues(new LinearRegressionModel(uid, model.coefficients, model.intercept))
      val (summaryModel, predictionColName) = lrModel.findSummaryModelAndPredictionCol()
      val trainingSummary = new LinearRegressionTrainingSummary(
        summaryModel.transform(dataset), predictionColName, $(labelCol), $(featuresCol),
        summaryModel, model.diagInvAtWA.toArray, model.objectiveHistory)

      lrModel.setSummary(Some(trainingSummary))
    }
  }

  private def trainWithConstantLabel(
      dataset: Dataset[_],
      instr: Instrumentation,
      numFeatures: Int,
      yMean: Double): LinearRegressionModel = {
    // If the rawYStd==0 and fitIntercept==true, then the intercept is yMean with
    // zero coefficient; as a result, training is not needed.
    // Also, if rawYStd==0 and yMean==0, all the coefficients are zero regardless of
    // the fitIntercept.
    if (yMean == 0.0) {
      instr.logWarning(s"Mean and standard deviation of the label are zero, so the " +
        s"coefficients and the intercept will all be zero; as a result, training is not " +
        s"needed.")
    } else {
      instr.logWarning(s"The standard deviation of the label is zero, so the coefficients " +
        s"will be zeros and the intercept will be the mean of the label; as a result, " +
        s"training is not needed.")
    }
    val coefficients = Vectors.sparse(numFeatures, Seq.empty)
    val intercept = yMean

    val model = copyValues(new LinearRegressionModel(uid, coefficients, intercept))
    // Handle possible missing or invalid prediction columns
    val (summaryModel, predictionColName) = model.findSummaryModelAndPredictionCol()

    val trainingSummary = new LinearRegressionTrainingSummary(
      summaryModel.transform(dataset), predictionColName, $(labelCol), $(featuresCol),
      model, Array(0.0), Array(0.0))

    model.setSummary(Some(trainingSummary))
  }

  private def createOptimizer(
      effectiveRegParam: Double,
      effectiveL1RegParam: Double,
      numFeatures: Int,
      featuresStd: Array[Double]) = {
    $(loss) match {
      case SquaredError =>
        if ($(elasticNetParam) == 0.0 || effectiveRegParam == 0.0) {
          new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
        } else {
          val standardizationParam = $(standardization)
          def effectiveL1RegFun = (index: Int) => {
            if (standardizationParam) {
              effectiveL1RegParam
            } else {
              // If `standardization` is false, we still standardize the data
              // to improve the rate of convergence; as a result, we have to
              // perform this reverse standardization by penalizing each component
              // differently to get effectively the same objective function when
              // the training dataset is not standardized.
              if (featuresStd(index) != 0.0) effectiveL1RegParam / featuresStd(index) else 0.0
            }
          }
          new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, effectiveL1RegFun, $(tol))
        }
      case Huber =>
        val dim = if ($(fitIntercept)) numFeatures + 2 else numFeatures + 1
        val lowerBounds = BDV[Double](Array.fill(dim)(Double.MinValue))
        // Optimize huber loss in space "\sigma > 0"
        lowerBounds(dim - 1) = Double.MinPositiveValue
        val upperBounds = BDV[Double](Array.fill(dim)(Double.MaxValue))
        new BreezeLBFGSB(lowerBounds, upperBounds, $(maxIter), 10, $(tol))
    }
  }

  private def trainImpl(
      instances: RDD[Instance],
      actualBlockSizeInMB: Double,
      yMean: Double,
      yStd: Double,
      featuresMean: Array[Double],
      featuresStd: Array[Double],
      initialValues: Vector,
      regularization: Option[L2Regularization],
      optimizer: FirstOrderMinimizer[BDV[Double], DiffFunction[BDV[Double]]]) = {
    val bcFeaturesMean = instances.context.broadcast(featuresMean)
    val bcFeaturesStd = instances.context.broadcast(featuresStd)

    val standardized = instances.mapPartitions { iter =>
      val inverseStd = bcFeaturesStd.value.map { std => if (std != 0) 1.0 / std else 0.0 }
      val func = StandardScalerModel.getTransformFunc(Array.empty, inverseStd, false, true)
      iter.map { case Instance(label, weight, vec) => Instance(label, weight, func(vec)) }
    }

    val maxMemUsage = (actualBlockSizeInMB * 1024L * 1024L).ceil.toLong
    val blocks = InstanceBlock.blokifyWithMaxMemUsage(standardized, maxMemUsage)
      .persist(StorageLevel.MEMORY_AND_DISK)
      .setName(s"training blocks (blockSizeInMB=$actualBlockSizeInMB)")

    val costFun = $(loss) match {
      case SquaredError =>
        val getAggregatorFunc = new BlockLeastSquaresAggregator(yStd, yMean, $(fitIntercept),
          bcFeaturesStd, bcFeaturesMean)(_)
        new RDDLossFunction(blocks, getAggregatorFunc, regularization, $(aggregationDepth))
      case Huber =>
        val getAggregatorFunc = new BlockHuberAggregator($(fitIntercept), $(epsilon))(_)
        new RDDLossFunction(blocks, getAggregatorFunc, regularization, $(aggregationDepth))
    }

    val states = optimizer.iterations(new CachedDiffFunction(costFun),
      initialValues.asBreeze.toDenseVector)

    /*
       Note that in Linear Regression, the objective history (loss + regularization) returned
       from optimizer is computed in the scaled space given by the following formula.
       <blockquote>
          $$
          L &= 1/2n||\sum_i w_i(x_i - \bar{x_i}) / \hat{x_i} - (y - \bar{y}) / \hat{y}||^2
               + regTerms \\
          $$
       </blockquote>
     */
    val arrayBuilder = mutable.ArrayBuilder.make[Double]
    var state: optimizer.State = null
    while (states.hasNext) {
      state = states.next()
      arrayBuilder += state.adjustedValue
    }

    blocks.unpersist()
    bcFeaturesMean.destroy()
    bcFeaturesStd.destroy()

    (if (state == null) null else state.x.toArray, arrayBuilder.result)
  }

  private def createModel(
      parameters: Array[Double],
      yMean: Double,
      yStd: Double,
      featuresMean: Array[Double],
      featuresStd: Array[Double]): LinearRegressionModel = {
    val numFeatures = featuresStd.length
    /*
       The coefficients are trained in the scaled space; we're converting them back to
       the original space.
     */
    val rawCoefficients = $(loss) match {
      case SquaredError => parameters.clone()
      case Huber => parameters.take(numFeatures)
    }

    var i = 0
    val len = rawCoefficients.length
    val multiplier = $(loss) match {
      case SquaredError => yStd
      case Huber => 1.0
    }
    while (i < len) {
      rawCoefficients(i) *= { if (featuresStd(i) != 0.0) multiplier / featuresStd(i) else 0.0 }
      i += 1
    }

    val intercept = if ($(fitIntercept)) {
      $(loss) match {
        case SquaredError =>
          /*
          The intercept of squared error in R's GLMNET is computed using closed form
          after the coefficients are converged. See the following discussion for detail.
          http://stats.stackexchange.com/questions/13617/how-is-the-intercept-computed-in-glmnet
          */
          yMean - dot(Vectors.dense(rawCoefficients), Vectors.dense(featuresMean))
        case Huber => parameters(numFeatures)
      }
    } else 0.0

    val scale = $(loss) match {
      case SquaredError => 1.0
      case Huber => parameters.last
    }

    val coefficients = Vectors.dense(rawCoefficients).compressed

    copyValues(new LinearRegressionModel(uid, coefficients, intercept, scale))
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}

@Since("1.6.0")
object LinearRegression extends DefaultParamsReadable[LinearRegression] {

  @Since("1.6.0")
  override def load(path: String): LinearRegression = super.load(path)

  /**
   * When using `LinearRegression.solver` == "normal", the solver must limit the number of
   * features to at most this number.  The entire covariance matrix X^T^X will be collected
   * to the driver. This limit helps prevent memory overflow errors.
   */
  @Since("2.1.0")
  val MAX_FEATURES_FOR_NORMAL_SOLVER: Int = WeightedLeastSquares.MAX_NUM_FEATURES

  /** String name for "auto". */
  private[regression] val Auto = "auto"

  /** String name for "normal". */
  private[regression] val Normal = "normal"

  /** String name for "l-bfgs". */
  private[regression] val LBFGS = "l-bfgs"

  /** Set of solvers that LinearRegression supports. */
  private[regression] val supportedSolvers = Array(Auto, Normal, LBFGS)

  /** String name for "squaredError". */
  private[regression] val SquaredError = "squaredError"

  /** String name for "huber". */
  private[regression] val Huber = "huber"

  /** Set of loss function names that LinearRegression supports. */
  private[regression] val supportedLosses = Array(SquaredError, Huber)
}


/** A writer for LinearRegression that handles the "internal" (or default) format */
private class InternalLinearRegressionModelWriter
  extends MLWriterFormat with MLFormatRegister {

  override def format(): String = "internal"
  override def stageName(): String = "org.apache.spark.ml.regression.LinearRegressionModel"

  private case class Data(intercept: Double, coefficients: Vector, scale: Double)

  override def write(path: String, sparkSession: SparkSession,
    optionMap: mutable.Map[String, String], stage: PipelineStage): Unit = {
    val instance = stage.asInstanceOf[LinearRegressionModel]
    val sc = sparkSession.sparkContext
    // Save metadata and Params
    DefaultParamsWriter.saveMetadata(instance, path, sc)
    // Save model data: intercept, coefficients, scale
    val data = Data(instance.intercept, instance.coefficients, instance.scale)
    val dataPath = new Path(path, "data").toString
    sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
  }
}

/** A writer for LinearRegression that handles the "pmml" format */
private class PMMLLinearRegressionModelWriter
  extends MLWriterFormat with MLFormatRegister {

  override def format(): String = "pmml"

  override def stageName(): String = "org.apache.spark.ml.regression.LinearRegressionModel"

  private case class Data(intercept: Double, coefficients: Vector)

  override def write(path: String, sparkSession: SparkSession,
    optionMap: mutable.Map[String, String], stage: PipelineStage): Unit = {
    val sc = sparkSession.sparkContext
    // Construct the MLLib model which knows how to write to PMML.
    val instance = stage.asInstanceOf[LinearRegressionModel]
    val oldModel = new OldLinearRegressionModel(instance.coefficients, instance.intercept)
    // Save PMML
    oldModel.toPMML(sc, path)
  }
}
