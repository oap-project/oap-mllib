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

package com.intel.oap.mllib.recommendation

import com.intel.oap.mllib.Utils

import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.spark313.{ALS => ALSSpark313}
import org.apache.spark.ml.recommendation.spark322.{ALS => ALSSpark322}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SPARK_VERSION, SparkException}

import scala.reflect.ClassTag

trait ALSShim extends Serializable with Logging {
  def train[ID: ClassTag]( // scalastyle:ignore
    ratings: RDD[Rating[ID]],
    rank: Int,
    numUserBlocks: Int,
    numItemBlocks: Int,
    maxIter: Int,
    regParam: Double,
    implicitPrefs: Boolean,
    alpha: Double,
    nonnegative: Boolean,
    intermediateRDDStorageLevel: StorageLevel,
    finalRDDStorageLevel: StorageLevel,
    checkpointInterval: Int,
    seed: Long)(implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])])
}

object ALSShim extends Logging {
  def create(): ALSShim = {
    logInfo(s"Loading ALS for Spark $SPARK_VERSION")
    val als = Utils.getSparkVersion() match {
      case "3.1.1" | "3.1.2" | "3.1.3" => new ALSSpark313()
      case "3.2.0" | "3.2.1" | "3.2.2" => new ALSSpark322()
      case _ => throw new SparkException(s"Unsupported Spark version $SPARK_VERSION")
    }
    als
  }
}
