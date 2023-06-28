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

package org.apache.spark.ml.clustering.spark321

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.clustering.{KMeansDALImpl, KMeansShim}

import org.apache.spark.annotation.Since
import org.apache.spark.ml.clustering.{KMeans => SparkKMeans, _}
import org.apache.spark.ml.functions.checkNonNegativeWeight
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.mllib.clustering.{DistanceMeasure, KMeans => MLlibKMeans, VectorWithNorm}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.mllib.linalg.VectorImplicits._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.storage.StorageLevel

/**
 * K-means clustering with support for k-means|| initialization proposed by Bahmani et al.
 *
 * @see <a href="https://doi.org/10.14778/2180912.2180915">Bahmani et al., Scalable k-means++.</a>
 */
@Since("1.5.0")
class KMeans @Since("1.5.0") (
    @Since("1.5.0") override val uid: String)
  extends SparkKMeans with KMeansShim {

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }

  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): KMeansModel = instrumented { instr =>
    transformSchema(dataset.schema, logging = true)

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, featuresCol, predictionCol, k, initMode, initSteps, distanceMeasure,
      maxIter, seed, tol, weightCol)

    val handleWeight = isDefined(weightCol) && $(weightCol).nonEmpty
    val w = if (handleWeight) {
      checkNonNegativeWeight(col($(weightCol)).cast(DoubleType))
    } else {
      lit(1.0)
    }
    val instances = dataset.select(DatasetUtils.columnToVector(dataset, getFeaturesCol), w)
      .rdd.map { case Row(point: Vector, weight: Double) => (point, weight) }

    val handlePersistence = (dataset.storageLevel == StorageLevel.NONE)

    val isPlatformSupported = Utils.checkClusterPlatformCompatibility(
      dataset.sparkSession.sparkContext)
    val useKMeansDAL = Utils.isOAPEnabled() && isPlatformSupported &&
      $(distanceMeasure) == "euclidean" && !handleWeight

    val model = if (useKMeansDAL) {
      trainWithDAL(instances, handlePersistence)
    } else {
      trainWithML(instances, handlePersistence)
    }

    val summary = new KMeansSummary(
      model.transform(dataset),
      $(predictionCol),
      $(featuresCol),
      $(k),
      model.parentModel.numIter,
      model.parentModel.trainingCost)

    model.setSummary(Some(summary))
    instr.logNamedValue("clusterSizes", summary.clusterSizes)

    model
  }

  private def trainWithDAL(instances: RDD[(Vector, Double)],
                           handlePersistence: Boolean): KMeansModel = instrumented { instr =>

    val sc = instances.sparkContext

    val executor_num = Utils.sparkExecutorNum(sc)
    val executor_cores = Utils.sparkExecutorCores()

    logInfo(s"KMeansDAL fit using $executor_num Executors")

    if (handlePersistence) {
      instances.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val initStartTime = System.nanoTime()

    val distanceMeasureInstance = DistanceMeasure.decodeFromString($(distanceMeasure))

    // Use MLlibKMeans to initialize centers
    val mllibKMeans = new MLlibKMeans()
      .setK($(k))
      .setInitializationMode($(initMode))
      .setInitializationSteps($(initSteps))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
      .setEpsilon($(tol))
      .setDistanceMeasure($(distanceMeasure))

    val dataWithNorm = instances.map {
      case (point: Vector, weight: Double) => new VectorWithNorm(point)
    }

    // Cache for init
    dataWithNorm.persist(StorageLevel.MEMORY_AND_DISK)

    val centersWithNorm = if ($(initMode) == "random") {
      mllibKMeans.initRandom(dataWithNorm)
    } else {
      mllibKMeans.initKMeansParallel(dataWithNorm, distanceMeasureInstance)
    }

    dataWithNorm.unpersist()

    val centers = centersWithNorm.map(_.vector)

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9

    val strInitMode = $(initMode)
    logInfo(f"Initialization with $strInitMode took $initTimeInSeconds%.3f seconds.")

    val inputData = instances.map {
      case (point: Vector, weight: Double) => point
    }

    if (handlePersistence) {
      inputData.persist(StorageLevel.MEMORY_AND_DISK)
      inputData.count()
    }

    val kmeansDAL = new KMeansDALImpl(getK, getMaxIter, getTol,
      DistanceMeasure.EUCLIDEAN, centers, executor_num, executor_cores)

    val parentModel = kmeansDAL.train(inputData)

    val model = copyValues(new KMeansModel(uid, parentModel).setParent(this))

    if (handlePersistence) {
      instances.unpersist()
    }

    model
  }

  private def trainWithML(instances: RDD[(Vector, Double)],
                          handlePersistence: Boolean): KMeansModel = instrumented { instr =>
      val oldVectorInstances = instances.map {
        case (point: Vector, weight: Double) => (OldVectors.fromML(point), weight)
      }
      val algo = new MLlibKMeans()
        .setK($(k))
        .setInitializationMode($(initMode))
        .setInitializationSteps($(initSteps))
        .setMaxIterations($(maxIter))
        .setSeed($(seed))
        .setEpsilon($(tol))
        .setDistanceMeasure($(distanceMeasure))
      val parentModel = algo.runWithWeight(oldVectorInstances, handlePersistence, Some(instr))
      val model = copyValues(new KMeansModel(uid, parentModel).setParent(this))
      model
    }
}

@Since("1.6.0")
object KMeans extends DefaultParamsReadable[KMeans] {

  @Since("1.6.0")
  override def load(path: String): KMeans = super.load(path)
}
