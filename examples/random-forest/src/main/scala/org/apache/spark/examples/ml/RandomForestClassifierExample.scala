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
import org.apache.spark.ml.{Pipeline, Transformer, linalg}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, LabeledPoint, StringIndexer, VectorIndexer, VectorIndexerModel}
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.util.{DatasetUtils, MetadataUtils}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{DataFrame, Row}

// $example off$
import org.apache.spark.sql.SparkSession

object RandomForestClassifierExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RandomForestClassifierExample")
      .config("spark.default.parallelism", 1)
      .getOrCreate()

    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val data = spark.read.format("libsvm").load("../data/sample_rf_csv_data.txt").toDF("label", "features")
    data.show(20,false)
    data.select("label","features").printSchema()
    val featuresRDD = data
      .select("label","features").rdd.map {
      case Row(label: Double ,feature: Vector) => new LabeledPoint(label, feature.toDense)
    }

    import spark.implicits._
    val df = featuresRDD.toDF("label", "features")
    df.show()
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(df)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(df)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = df.randomSplit(Array(1, 0))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setNumTrees(1)
      .setSeed(777L)
      .setBootstrap(false)

    rf.getClass.read
    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labelsArray(0))

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    trainingData.show()
    println(s"row account ${trainingData.count()}")
    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(trainingData)

    // Select example rows to display.
    println(s"Select example rows to display")
    predictions.show()
    println(s"Select example rows to display end")

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")


    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"tree length: ${rfModel.trees.length}")
    for ( m <- rfModel.trees) {
      println(s"prediction : ${m.rootNode.prediction}")
      println(s"impurity: ${m.rootNode.impurity}")
      println(s"oldnode: ${m.rootNode.toString}")
    }
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
