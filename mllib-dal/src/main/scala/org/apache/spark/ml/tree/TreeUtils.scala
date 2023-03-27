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
package org.apache.spark.ml.tree

import com.intel.oap.mllib.classification.{LearningNode => LearningNodeDAL}

import org.apache.spark.mllib.tree.impurity.GiniCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats


object TreeUtils {
  def buildTreeDFS(nodes : java.util.ArrayList[LearningNodeDAL]) : LearningNode = {
    val currentLevel = nodes.get(0).level ;
    if (nodes.isEmpty) {
      return null
    }
    val impurityCalculator = new GiniCalculator(nodes.get(0).probability, nodes.get(0).sampleCount)
    val impurityStats = new ImpurityStats(0, nodes.get(0).impurity, impurityCalculator, null, null)
    val rootNode = LearningNode.apply(0, nodes.get(0).isLeaf, impurityStats)
    buildTreeDF(rootNode, nodes, 1, currentLevel)
    calculateGainAndImpurityStats(rootNode)
    traverseDFS(rootNode)
    rootNode
  }
  private def traverseDFS(rootNode: LearningNode): Unit = {
    println(s"split is : ${rootNode.split.nonEmpty}; " +
      s"leftChild is : ${rootNode.leftChild.nonEmpty} ;" +
      s"rightChild is : ${rootNode.rightChild.nonEmpty} ;" +
      s"state is : ${rootNode.stats.toString()}")
    if (!rootNode.leftChild.isEmpty) {
      traverseDFS(rootNode.leftChild.getOrElse(null))
    }
    if (!rootNode.rightChild.isEmpty) {
      traverseDFS(rootNode.rightChild.getOrElse(null))
    }
  }

  private def buildTreeDF(parentNode : LearningNode,
                          nodes : java.util.ArrayList[LearningNodeDAL],
                          index : Int,
                          parentLevel : Int) : Unit = {
    if (nodes.get(index) == null) {
      return
    }
    if (parentLevel < nodes.get(index).level) {
      val impurityCalculator = new GiniCalculator(nodes.get(index).probability,
                                                  nodes.get(index).sampleCount)
      val impurityStats = new ImpurityStats(0,
                                            nodes.get(index).impurity,
                                            impurityCalculator,
                                            null,
                                            null)
      val childNode = LearningNode.apply(index, nodes.get(index).isLeaf, impurityStats)
      parentNode.leftChild = Some(childNode)
      buildTreeDF(childNode, nodes, index + 1, nodes.get(index).level)
    } else if (parentLevel == nodes.get(index).level) {
      val impurityCalculator = new GiniCalculator(nodes.get(index).probability,
                                                  nodes.get(index).sampleCount)
      val impurityStats = new ImpurityStats(0,
                                            nodes.get(index).impurity,
                                            impurityCalculator,
                                            null,
                                            null)
      val childNode = LearningNode.apply(index, nodes.get(index).isLeaf, impurityStats)
      parentNode.rightChild = Some(childNode)
      buildTreeDF(childNode, nodes, index + 1, nodes.get(index).level-1)
    }
  }

  private def calculateGainAndImpurityStats (parentNode : LearningNode): Unit = {
    val impurity: Double = parentNode.stats.impurity
    val left = parentNode.leftChild.getOrElse(null)
    val right = parentNode.rightChild.getOrElse(null)
    val leftImpurityCalculator = if (left != null) {
      calculateGainAndImpurityStats(left)
      left.stats.impurityCalculator
    } else {
      null
    }
    val rightImpurityCalculator = if (right != null) {
      calculateGainAndImpurityStats(right)
      right.stats.impurityCalculator
    } else {
      null
    }

    val (leftCount, leftImpurity) = if (leftImpurityCalculator != null) {
      val count = if (leftImpurityCalculator.stats == null ) {
        0.0
      } else {
        leftImpurityCalculator.stats.count(_ != 0.0)
      }
      val sum = if (leftImpurityCalculator.stats == null ) {
        0.0
      } else {
        leftImpurityCalculator.stats.sum
      }
      val calculateImpurity = left.stats.impurity - sum
      (count, calculateImpurity)
    } else {
      (0.0, 0.0)
    }
    val (rightCount, rightImpurity) = if (rightImpurityCalculator != null) {
      val count = if (rightImpurityCalculator.stats == null ) {
        0.0
      } else {
        rightImpurityCalculator.stats.count(_ != 0.0)
      }
      val sum = if (rightImpurityCalculator.stats == null ) {
        0.0
      } else {
        rightImpurityCalculator.stats.sum
      }
      val calculateImpurity = right.stats.impurity - sum
      (count, calculateImpurity)
    } else {
      (0.0, 0.0)
    }
    val totalCount: Double = leftCount + rightCount
    val leftWeight: Double = leftCount / totalCount.toDouble
    val rightWeight: Double = rightCount/ totalCount.toDouble

    val gain: Double = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity
    parentNode.stats = new ImpurityStats(gain,
      impurity,
      parentNode.stats.impurityCalculator,
      leftImpurityCalculator,
      rightImpurityCalculator)
  }
}
