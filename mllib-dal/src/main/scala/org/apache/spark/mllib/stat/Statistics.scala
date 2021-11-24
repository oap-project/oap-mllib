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

package org.apache.spark.mllib.stat

import com.intel.oap.mllib.stat.{CorrelationShim, SummarizerShim}
import org.apache.spark.annotation.Since
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrix, Vector}
import org.apache.spark.mllib.stat.correlation.Correlations




/**
  * API for statistical functions in MLlib.
  */
@Since("1.1.0")
object Statistics {

  /**
    * Computes column-wise summary statistics for the input RDD[Vector].
    *
    * @param X an RDD[Vector] for which column-wise summary statistics are to be computed.
    * @return [[MultivariateStatisticalSummary]] object containing column-wise summary statistics.
    */
  @Since("1.1.0")
  def colStats(X: RDD[Vector]): MultivariateStatisticalSummary = {
    val shim = SummarizerShim.create()
    shim.colStats(X)
  }
  /**
    * Compute the Pearson correlation matrix for the input RDD of Vectors.
    * Columns with 0 covariance produce NaN entries in the correlation matrix.
    *
    * @param X an RDD[Vector] for which the correlation matrix is to be computed.
    * @return Pearson correlation matrix comparing columns in X.
    */
  @Since("1.1.0")
  def corr(X: RDD[Vector]): Matrix = Correlations.corrMatrix(X)

  /**
    * Compute the correlation matrix for the input RDD of Vectors using the specified method.
    * Methods currently supported: `pearson` (default), `spearman`.
    *
    * @param X an RDD[Vector] for which the correlation matrix is to be computed.
    * @param method String specifying the method to use for computing correlation.
    *               Supported: `pearson` (default), `spearman`
    * @return Correlation matrix comparing columns in X.
    *
    * @note For Spearman, a rank correlation, we need to create an RDD[Double] for each column
    * and sort it in order to retrieve the ranks and then join the columns back into an RDD[Vector],
    * which is fairly costly. Cache the input RDD before calling corr with `method = "spearman"` to
    * avoid recomputing the common lineage.
    */
  @Since("1.1.0")
  def corr(X: RDD[Vector], method: String): Matrix = Correlations.corrMatrix(X, method)

  /**
    * Compute the Pearson correlation for the input RDDs.
    * Returns NaN if either vector has 0 variance.
    *
    * @param x RDD[Double] of the same cardinality as y.
    * @param y RDD[Double] of the same cardinality as x.
    * @return A Double containing the Pearson correlation between the two input RDD[Double]s
    *
    * @note The two input RDDs need to have the same number of partitions and the same number of
    * elements in each partition.
    */
  @Since("1.1.0")
  def corr(x: RDD[Double], y: RDD[Double]): Double = Correlations.corr(x, y)

  /**
    * Java-friendly version of `corr()`
    */
  @Since("1.4.1")
  def corr(x: JavaRDD[java.lang.Double], y: JavaRDD[java.lang.Double]): Double =
    corr(x.rdd.asInstanceOf[RDD[Double]], y.rdd.asInstanceOf[RDD[Double]])

  /**
    * Compute the correlation for the input RDDs using the specified method.
    * Methods currently supported: `pearson` (default), `spearman`.
    *
    * @param x RDD[Double] of the same cardinality as y.
    * @param y RDD[Double] of the same cardinality as x.
    * @param method String specifying the method to use for computing correlation.
    *               Supported: `pearson` (default), `spearman`
    * @return A Double containing the correlation between the two input RDD[Double]s using the
    *         specified method.
    *
    * @note The two input RDDs need to have the same number of partitions and the same number of
    * elements in each partition.
    */
  @Since("1.1.0")
  def corr(x: RDD[Double], y: RDD[Double], method: String): Double = Correlations.corr(x, y, method)

  /**
    * Java-friendly version of `corr()`
    */
  @Since("1.4.1")
  def corr(x: JavaRDD[java.lang.Double], y: JavaRDD[java.lang.Double], method: String): Double =
    corr(x.rdd.asInstanceOf[RDD[Double]], y.rdd.asInstanceOf[RDD[Double]], method)

}
