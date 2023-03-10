/*******************************************************************************
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
 *******************************************************************************/

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "com_intel_oap_mllib_classification_RandomForestClassifierDALImpl.h"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;
const std::shared_ptr<std::map<std::int64_t, std::vector<std::string>>> treeForest = std::make_shared<std::map<std::int64_t, std::vector<std::string>>>();
const char* LEAFNODE_CLASS_NAME = "org.apache.spark.ml.tree.LeafNode";
const char* INTERNALNODE_CLASS_NAME = "org.apache.spark.ml.tree.InternalNode";
const char* SPLIT_CLASS_NAME = "org/apache/spark/ml/tree/Split";
const char* IMPURITY_CALCULATOR_CLASS_NAME = "org.apache.spark.mllib.tree.impurity.ImpurityCalculator";

// Define the LearningNode struct
typedef struct {
    int id;
    int level;
    LearningNode leftChild;
    LearningNode rightChild;
    LearningNode split;
    bool isLeaf;
    jobject stats;
} LearningNode;

struct trees_wrap{

    std::shared_ptr<std::vector<jobejct>> treesVector;
    JNIEnv *treeenv,
    int i = 0;
    trees_wrap(JNIEnv *env, const std::shared_ptr<std::vector<jobejct>> &vector) {
        treesVector = vector;
        treeenv = env;
    }
    bool operator()(const df::leaf_node_info<df::task::classification>& info) {
        std::string str;
        str.append("leaf");
        str.append("|" );
        str.append(to_string(info.get_level()));
        str.append("|" );
        str.append(to_string(info.get_response()));
        str.append("|" );
        str.append(to_string(info.get_impurity()));
        str.append("|" );
        str.append(to_string(info.get_sample_count()));

        treesVector->push_back(createLearningNode(env, info));
        std::cout << str << std::endl;

        i++;
        return true;

    bool operator()(const df::split_node_info<df::task::classification>& info) {
        std::string str;
        str.append("split");
        str.append("|" );
        str.append(to_string(info.get_level()));
        str.append("|" );
        str.append(to_string(info.get_feature_index()));
        str.append("|" );
        str.append(to_string(info.get_feature_value()));
        str.append("|" );
        str.append(to_string(info.get_impurity()));
        str.append("|" );
        str.append(to_string(info.get_sample_count()));

        treesVector->push_back(createLearningNode(env, info));

        std::cout << str << std::endl;
        i++;
        return true;
    }
};

template <typename Task>
jobject print_model(JNIEnv *env, const df::model<Task>& m) {
   jclass randomForestClassificationModelClass = env->FindClass("org.apache.spark.ml.classification.RandomForestClassificationModel");
   jclass decisionTreeClassificationModelCLass = env->FindClass("org.apache.spark.ml.classification.DecisionTreeClassificationModel");



    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;

        std::shared_ptr<std::vector<jobejct>> myVec = std::make_shared<std::vector<jobejct>>();
        m.traverse_depth_first(i, trees_wrap{env, myVec});
        for(int i=0; i < myVec.size(); i++) {

        }

    }
}

jobject buildTree(JNIEnv* env, std::shared_ptr<std::vector<jobejct>> nodes, jint level, jint nodeIndex) {
    // Get the current node from the array
    jobject currentNode = nodes[index];
    // Recursively build the child nodes if the current node is not a leaf
    if (!isLeaf) {
        jobject leftNode = buildTree(env, clazz, nodes, index + 1);
        jobject rightNode = buildTree(env, clazz, nodes, index + 2);

        // Set the child nodes of the current node
        env->SetObjectField(currentNode, env->GetFieldID(learningNodeClass, "leftChild", "Lcom/example/LearningNode;"), leftNode);
        env->SetObjectField(currentNode, env->GetFieldID(learningNodeClass, "rightChild", "Lcom/example/LearningNode;"), rightNode);
    }
}

jobejct createSplitNode(JNIEnv* env, const df::split_node_info<df::task::classification>& info, int nodeIndex) {
}

jobject createLeafNode(JNIEnv* env, const df::leaf_node_info<df::task::classification>& info, int nodeIndex) {

    // 获取类和构造函数
    jclass leafNodeClass = env->FindClass(LEAFNODE_CLASS_NAME);
    if(leafNodeClass == NULL){
        return;
    }
    jmethodID constructor = env->GetMethodID(leafNodeClass, "<init>", "(Ljava/lang/Long;Ljava/lang/Long;Lorg.apache.spark.mllib.tree.impurity.ImpurityCalculator)V");
    jfieldID idField = env->GetFieldID(learningNodeClass, "id", "I");
    jfieldID leftChildField = env->GetFieldID(learningNodeClass, "leftChild", "Lscala/Option;");
    jfieldID rightChildField = env->GetFieldID(learningNodeClass, "rightChild", "Lscala/Option;");
    jfieldID splitField = env->GetFieldID(learningNodeClass, "split", "Lscala/Option;");
    jfieldID isLeafField = env->GetFieldID(learningNodeClass, "isLeaf", "Z");
    jfieldID statsField = env->GetFieldID(learningNodeClass, "stats", "Lorg/apache/spark/ml/tree/ImpurityStats;");

    jobject leftChildObj = env->GetObjectField(node, leftChildId);
    jobject rightChildObj = env->GetObjectField(node, rightChildId);
    jboolean isLeaf = env->GetBooleanField(node, isLeafId);
    jobject splitObj = env->GetObjectField(node, splitId);
    jobject statsObj = env->GetObjectField(node, statsId);

    // set Id
    env->SetIntField(node, idField,
                             nodeIndex);
    // set isleaf
    env->SetBooleanField(node, isLeafField,
                            typeid(info) ==typeid(df::leaf_node_info) ? true : false);
    // set leftChild
    env->SetIntField(node, leftChildField,
                                 nullptr);
    // set rightChild
    env->SetIntField(node, rightChildField,
                                 nullptr);
    // set split
    if(!typeid(info) ==typeid(df::leaf_node_info)){
        env->SetIntField(node, splitField,
                                 info.get_feature_index());
    }

    // 创建ImpurityStats对象
    jclass statsCls = env->FindClass(IMPURITY_STATS_CLASS_NAME);
    // Get Field references
    jfieldID gainField = env->GetFieldID(statsCls, "gain", "D"));
    jfieldID impurityField = env->GetFieldID(statsCls, "impurity", "D"));
    jfieldID impurityField = env->GetFieldID(statsCls, "impurityCalculator", "Lorg.apache.spark.mllib.tree.impurity.ImpurityCalculator"));
    jfieldID impurityField = env->GetFieldID(statsCls, "leftImpurityCalculator", "Lorg.apache.spark.mllib.tree.impurity.ImpurityCalculator"));
    jfieldID impurityField = env->GetFieldID(statsCls, "rightImpurityCalculator", "Lorg.apache.spark.mllib.tree.impurity.ImpurityCalculator"));

    env->SetIntField(statsObj, gainField,
                             0.0));
    env->SetIntField(statsObj, impurityField,
                         info.get_impurity());


    jobject nodeObj = env->NewObject(learningNodeClass, constructor);
    // 释放对象引用

    return nodeObj;
}

static jobject doRFClassifierOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabFeature,
                                           jlong pNumTabLabel, jint executorNum,
                                           jint computeDeviceOrdinal, jint classCount, jint treeCount,
                                           jint minObservationsLeafNode, jint minObservationsSplitNode,
                                           jdouble minWeightFractionLeafNode, jdouble minImpurityDecreaseSplitNode,
                                           const ccl::string &ipPort,
                                           jobject resultObj) {
   std::cout << "oneDAL (native): compute start , rankid = " << rankId
                 << "; device = " << ComputeDeviceString[computeDeviceOrdinal]
                 << std::endl;
   const bool isRoot = (rankId == ccl_root);
   ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
   homogen_table hFeaturetable =
       *reinterpret_cast<const homogen_table *>(pNumTabFeature);
   homogen_table hLabeltable =
       *reinterpret_cast<const homogen_table *>(pNumTabLabel);
   const auto df_desc =
        df::descriptor<float, df::method::hist, df::task::classification>{}
            .set_class_count(classCount)
            .set_tree_count(treeCount)
            .set_features_per_node(htable.get_column_count())
            .set_min_observations_in_leaf_node(minObservationsLeafNode)
            .set_min_observations_in_split_node(minObservationsSplitNode)
            .set_min_weight_fraction_in_leaf_node(minWeightFractionLeafNode)
            .set_min_impurity_decrease_in_split_node(minImpurityDecreaseSplitNode)
            .set_error_metric_mode(df::error_metric_mode::out_of_bag_error)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_infer_mode(df::infer_mode::class_responses |
                            df::infer_mode::class_probabilities)
            .set_voting_mode(df::voting_mode::weighted);
    auto queue = getQueue(device);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
            queue, executorNum, rankId, ipPort);
    const auto result_train =
        dal::preview::train(comm, df_desc, hFeaturetable, hLabeltable);
    print_model(result_train.get_model());
    const auto result_infer =
        dal::preview::infer(comm, df_desc, result_train.get_model(), hFeaturetable);

    if (comm.get_rank() == 0) {
        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;
        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n" << result_infer.get_probabilities() << std::endl;
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID predictionNumericTableField = env->GetFieldID(clazz, "predictionNumericTable", "J");
        jfieldID probabilitiesNumericTableField =
            env->GetFieldID(clazz, "probabilitiesNumericTable", "J");
        // Set iteration num for result
        env->SetIntField(resultObj, predictionNumericTableField,
                         result_infer.get_responses());
        // Set cost for result
        env->SetDoubleField(resultObj, probabilitiesNumericTableField,
                            result_infer.get_probabilities());
        return (jlong)*treeForest;
    }else {
        return (jlong)0;
    }

}

/*
 * Class:     com_intel_oap_mllib_classification_RandomForestClassifierDALImpl
 * Method:    cRFClassifierTrainDAL
 * Signature: (JIIIIIIIIDDLjava/lang/String;Lcom/intel/oap/mllib/classification/RandomForestResult;)J
 */
JNIEXPORT jobject JNICALL Java_com_intel_oap_mllib_classification_RandomForestClassifierDALImpl_cRFClassifierTrainDAL
  (JNIEnv *env, jobject obj, jlong pNumTabFeature, jlong pNumTabLabel, jint executorNum,
  jint computeDeviceOrdinal, jint classCount, jint rankId, jint treeCount,
  jint minObservationsLeafNode, jint minObservationsSplitNode,
  jdouble minWeightFractionLeafNode, jdouble minImpurityDecreaseSplitNode, jstring ipPort,
  jobject resultObj) {
      std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
      const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
      std::string ipPortStr = std::string(ipPortPtr);
      jlong ret = 0L;
      ret = doRFClassifierOneAPICompute(
          env, rankId, pNumTabFeature, pNumTabLabel, executorNum, computeDeviceOrdinal, classCount,
          treeCount, minObservationsLeafNode, minObservationsSplitNode,
          minWeightFractionLeafNode, minImpurityDecreaseSplitNode,
          ipPortPtr, resultObj);
      env->ReleaseStringUTFChars(ipPort, ipPortPtr);
      return ret;
  }
