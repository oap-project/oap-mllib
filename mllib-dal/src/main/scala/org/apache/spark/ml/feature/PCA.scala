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

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.mllib.feature
import org.apache.spark.mllib.feature.{PCAModel => OldPCAModel}
import org.apache.spark.mllib.linalg.{DenseMatrix => OldDenseMatrix, Vectors => OldVectors}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.VersionUtils.majorVersion
import com.intel.oap.mllib.Utils
import com.intel.oap.mllib.feature.{PCADALImpl, PCAShim}

/**
 * PCA trains a model to project vectors to a lower dimensional space of the top `PCA!.k`
 * principal components.
 */
@Since("1.5.0")
class PCA @Since("1.5.0") (
    @Since("1.5.0") override val uid: String)
  extends Estimator[PCAModel] with PCAParams with DefaultParamsWritable {

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("pca"))

  /** @group setParam */
  @Since("1.5.0")
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  @Since("1.5.0")
  def setK(value: Int): this.type = set(k, value)

  /**
   * Computes a [[PCAModel]] that contains the principal components of the input vectors.
   */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): PCAModel = {
    val pca = PCAShim.create(uid)
      .setInputCol($(inputCol))
      .setOutputCol($(outputCol))
      .setK($(k))
    pca.fit(dataset)
  }

  @Since("1.5.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): PCA = defaultCopy(extra)
}

@Since("1.6.0")
object PCA extends DefaultParamsReadable[PCA] {

  @Since("1.6.0")
  override def load(path: String): PCA = super.load(path)
}
