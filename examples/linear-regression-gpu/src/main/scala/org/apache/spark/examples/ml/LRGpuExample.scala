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

package org.apache.spark.examples.ml

// scalastyle:off println

// $example on$
//import org.apache.spark.ml.regression.LinearRegression
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating k-means clustering. 
 * Slightly modified from MLlib KMeansExample to add data file path as input parameter
 */
object LRExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()

    if (args.length != 1) {
      println("Require data file path as input parameter")
      sys.exit(1)
    }

    // $example on$
    // Loads data.
    val dataset = spark.read.format("libsvm").load(args(0))
    dataset.cache()

    // Trains a k-means model.
    println("seems good0")
    //val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    // Fit the model
    //val lrModel = lr.fit(dataset)

    // Print the coefficients and intercept for linear regression
    //println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the dataset and print out some metrics
    //val trainingSummary = lrModel.summary
    //println(s"numIterations: ${trainingSummary.totalIterations}")
    //println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    //trainingSummary.residuals.show()
    //println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    //println(s"r2: ${trainingSummary.r2}")

    spark.stop()
  }
}
// scalastyle:on println
