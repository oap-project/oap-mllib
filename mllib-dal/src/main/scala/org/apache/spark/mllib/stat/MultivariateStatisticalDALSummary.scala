/*
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.stat

import org.apache.spark.mllib.linalg.Vector

class MultivariateStatisticalDALSummary (
              val meanVector: Vector,
              val varianceVector: Vector,
              val maxVector: Vector,
              val minVector: Vector)
  extends MultivariateStatisticalSummary with Serializable {

  /*
   * Sample mean vector.
   */
  override def mean: Vector = {
    meanVector
  }

  /*
   * Sample variance vector. Should return a zero vector if the sample size is 1.
   */
  override def variance: Vector = {
    varianceVector
  }

  /*
   * Sample size.
   */
  override def count: Long = 0

  /*
   * Sum of weights.
   */
  override def weightSum: Double = 0.0

  /*
   * Number of nonzero elements (including explicitly presented zero values) in each column.
   */
  override def numNonzeros: Vector = null

  /*
   * Maximum value of each column.
   */
  override def max: Vector = {
    maxVector
  }

  /*
   * Minimum value of each column.
   */
  override def min: Vector = {
    minVector
  }

  /*
   * Euclidean magnitude of each column
   */
  override def normL2: Vector = null

  /*
   * L1 norm of each column
   */
  override def normL1: Vector = null

}
