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

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
// $example off$
import org.apache.spark.sql.SparkSession

object NaiveBayesExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("NaiveBayesExample")
      .getOrCreate()

    if (args.length != 1) {
      println("Require data file path as input parameter")
      sys.exit(1)
    }

    // $example on$
    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load(args(0)).toDF("label", "featurescolumn")
    println(data.schema.fieldNames.toList)
    // Train a NaiveBayes model.
    val model = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("featurescolumn")
      .fit(data)

    println(model.getFeaturesCol)
    spark.stop()
  }
}
// scalastyle:on println
