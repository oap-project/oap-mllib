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

package org.apache.spark.ml.feature.spark320

import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.feature.{PCADALImpl, PCAShim}

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.{PCA => SparkPCA, _}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.feature.{PCAModel => OldPCAModel}
import org.apache.spark.mllib.linalg.{Vectors => OldVectors}
import org.apache.spark.sql._

/**
 * PCA trains a model to project vectors to a lower dimensional space of the top `PCA!.k`
 * principal components.
 */
@Since("1.5.0")
class PCA @Since("1.5.0")(@Since("1.5.0") override val uid: String)
    extends SparkPCA
    with PCAShim {

  override def initShim(params: ParamMap): Unit = {
    params.toSeq.foreach { paramMap.put(_) }
  }

  /**
   * Computes a [[PCAModel]] that contains the principal components of the input vectors.
   */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): PCAModel = {
    transformSchema(dataset.schema, logging = true)
    val input = dataset.select($(inputCol)).rdd
    val inputVectors = input.map {
      case Row(v: Vector) => v
    }

    val numFeatures = inputVectors.first().size
    require($(k) <= numFeatures, s"source vector size $numFeatures must be no less than k=$k")

    val isPlatformSupported =
      Utils.checkClusterPlatformCompatibility(dataset.sparkSession.sparkContext)

    // Call oneDAL Correlation PCA implementation when numFeatures < 65535 and fall back otherwise
    val parentModel = if (numFeatures < 65535 && Utils.isOAPEnabled() && isPlatformSupported) {
      val executor_num = Utils.sparkExecutorNum(dataset.sparkSession.sparkContext)
      val executor_cores = Utils.sparkExecutorCores()
      val pca = new PCADALImpl(k = $(k), executor_num, executor_cores)
      val pcaDALModel = pca.train(inputVectors)
      new OldPCAModel(pcaDALModel.k, pcaDALModel.pc, pcaDALModel.explainedVariance)
    } else {
      val inputOldVectors = inputVectors.map {
        case v: Vector => OldVectors.fromML(v)
      }
      val pca = new feature.PCA(k = $(k))
      val pcaModel = pca.fit(inputOldVectors)
      pcaModel
    }
    copyValues(
      new PCAModel(uid, parentModel.pc.asML, parentModel.explainedVariance.asML)
        .setParent(this))
  }
}
