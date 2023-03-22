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
    rootNode
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
}
