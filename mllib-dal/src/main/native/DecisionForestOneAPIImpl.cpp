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
#include <iterator>
#include <map>
#include <string>
#include <vector>

#ifdef CPU_GPU_PROFILE
#include "Common.hpp"

#include "OneCCL.h"
#include "com_intel_oap_mllib_classification_RandomForestClassifierDALImpl.h"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
namespace df = oneapi::dal::decision_forest;

// Define the LearningNode struct
struct LearningNode {
    constexpr LearningNode(){};
    int level = 0;
    double impurity = 0.0;
    int splitIndex = 0;
    double splitValue = 0.0;
    bool isLeaf = false;
    std::unique_ptr<double[]> probability = nullptr;
    int sampleCount = 0;
};
const std::shared_ptr<
    std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>>
    treeForest = std::make_shared<
        std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>>();

LearningNode convertsplitToLearningNode(
    const df::split_node_info<df::task::classification> &info,
    const int classCount) {
    LearningNode splitNode;
    splitNode.isLeaf = false;
    splitNode.level = info.get_level();
    splitNode.splitIndex = info.get_feature_index();
    splitNode.splitValue = info.get_feature_value();
    splitNode.impurity = info.get_impurity();
    splitNode.sampleCount = info.get_sample_count();
    std::unique_ptr<double[]> arr(new double[classCount]);
    for (std::int64_t index_class = 0; index_class < classCount;
         ++index_class) {
        arr[index_class] = 0.0;
    }
    splitNode.probability = std::move(arr);
    return splitNode;
}

LearningNode convertleafToLearningNode(
    const df::leaf_node_info<df::task::classification> &info,
    const int classCount) {
    LearningNode leafNode;
    leafNode.isLeaf = true;
    leafNode.level = info.get_level();
    leafNode.impurity = info.get_impurity();
    leafNode.sampleCount = info.get_sample_count();
    std::unique_ptr<double[]> arr(new double[classCount]);
    for (std::int64_t index_class = 0; index_class < classCount;
         ++index_class) {
        arr[index_class] = info.get_probability(index_class);
    }
    leafNode.probability = std::move(arr);
    return leafNode;
}

struct collect_nodes {
    std::shared_ptr<std::vector<LearningNode>> treesVector;
    jint classCount;
    collect_nodes(const std::shared_ptr<std::vector<LearningNode>> &vector,
                  int count) {
        treesVector = vector;
        classCount = count;
    }
    bool operator()(const df::leaf_node_info<df::task::classification> &info) {
        std::string str;
        str.append("leaf");
        str.append("|");
        str.append(to_string(info.get_level()));
        str.append("|");
        str.append(to_string(info.get_response()));
        str.append("|");
        str.append(to_string(info.get_impurity()));
        str.append("|");
        str.append(to_string(info.get_sample_count()));

        treesVector->push_back(convertleafToLearningNode(info, classCount));
        std::cout << str << std::endl;

        i++;
        return true;
    }

    bool operator()(const df::split_node_info<df::task::classification> &info) {
        std::string str;
        str.append("split");
        str.append("|");
        str.append(to_string(info.get_level()));
        str.append("|");
        str.append(to_string(info.get_feature_index()));
        str.append("|");
        str.append(to_string(info.get_feature_value()));
        str.append("|");
        str.append(to_string(info.get_impurity()));
        str.append("|");
        str.append(to_string(info.get_sample_count()));

        treesVector->push_back(convertsplitToLearningNode(info, classCount));

        std::cout << str << std::endl;
        i++;
        return true;
    }
};

template <typename Task>
void collect_model(
    JNIEnv *env, const df::model<Task> &m, const jint classCount,
    const std::shared_ptr<
        std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>>
        &treeForest) {
    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;
        std::shared_ptr<std::vector<LearningNode>> myVec =
            std::make_shared<std::vector<LearningNode>>();
        m.traverse_depth_first(i, collect_nodes{myVec, classCount});
        treeForest->insert({i, myVec});
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

jobject convertJavaMap(
    JNIEnv *env,
    const std::shared_ptr<
        std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>>
        map,
    const jint classCount) {
    std::cout << "convert map to HashMap " << std::endl;
    // Create a new Java HashMap object
    jclass mapClass = env->FindClass("java/util/HashMap");
    jmethodID mapConstructor = env->GetMethodID(mapClass, "<init>", "()V");
    jobject jMap = env->NewObject(mapClass, mapConstructor);

    // Get the List class and its constructor
    jclass listClass = env->FindClass("java/util/ArrayList");
    jmethodID listConstructor = env->GetMethodID(listClass, "<init>", "()V");

    // Get the LearningNode class and its constructor
    jclass learningNodeClass =
        env->FindClass("com/intel/oap/mllib/classification/LearningNode");
    jmethodID learningNodeConstructor =
        env->GetMethodID(learningNodeClass, "<init>", "()V");

    // Iterate over the C++ map and add each entry to the Java map
    for (const auto &entry : *map) {
        std::cout
            << "Iterate over the C++ map and add each entry to the Java map"
            << std::endl;
        // Create a new Java ArrayList to hold the LearningNode objects
        jobject jList = env->NewObject(listClass, listConstructor);

        // Iterate over the LearningNode objects and add each one to the
        // ArrayList
        for (const auto &node : *entry.second) {
            jobject jNode =
                env->NewObject(learningNodeClass, learningNodeConstructor);

            // Set the fields of the Java LearningNode object using JNI
            // functions
            jfieldID levelField =
                env->GetFieldID(learningNodeClass, "level", "I");
            env->SetIntField(jNode, levelField, node.level);

            jfieldID impurityField =
                env->GetFieldID(learningNodeClass, "impurity", "D");
            env->SetDoubleField(jNode, impurityField, node.impurity);

            jfieldID splitIndexField =
                env->GetFieldID(learningNodeClass, "splitIndex", "I");
            env->SetIntField(jNode, splitIndexField, node.splitIndex);

            jfieldID splitValueField =
                env->GetFieldID(learningNodeClass, "splitValue", "D");
            env->SetDoubleField(jNode, splitValueField, node.splitValue);

            jfieldID isLeafField =
                env->GetFieldID(learningNodeClass, "isLeaf", "Z");
            env->SetBooleanField(jNode, isLeafField, node.isLeaf);

            // Convert the probability array
            if (node.probability != nullptr) {
                std::cout << "convertJavaMap node.probability  = "
                          << node.probability.get()[0] << std::endl;
                jfieldID probabilityField =
                    env->GetFieldID(learningNodeClass, "probability", "[D");
                jdoubleArray jProbability = env->NewDoubleArray(classCount);
                // call the native method to get the values
                jdouble *elements =
                    env->GetDoubleArrayElements(jProbability, nullptr);
                // Do something with the array elements
                for (int i = 0; i < classCount; i++) {
                    elements[i] = node.probability.get()[i];
                }
                env->SetDoubleArrayRegion(jProbability, 0, classCount,
                                          elements);
                env->SetObjectField(jNode, probabilityField, jProbability);
            }

            jfieldID sampleCountField =
                env->GetFieldID(learningNodeClass, "sampleCount", "I");
            env->SetIntField(jNode, sampleCountField, node.sampleCount);

            // Add the LearningNode object to the ArrayList
            jmethodID listAdd =
                env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
            env->CallBooleanMethod(jList, listAdd, jNode);
        }

        // Add the ArrayList to the Java map
        jmethodID mapPut = env->GetMethodID(
            mapClass, "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        std::cout << "convertJavaMap tree id  = " << entry.first << std::endl;
        // Create a new Integer object with the value key
        jobject jKey = env->NewObject(
            env->FindClass("java/lang/Integer"), // Find the Integer class
            env->GetMethodID(env->FindClass("java/lang/Integer"), "<init>",
                             "(I)V"), // Get the constructor method
            (jint) static_cast<jint>(entry.first));
        env->CallObjectMethod(jMap, mapPut, jKey, jList);
    }
    std::cout << "convert map to HashMap end " << std::endl;

    return jMap;
}

static jobject doRFClassifierOneAPICompute(
    JNIEnv *env, jlong pNumTabFeature, jlong pNumTabLabel, jint executorNum,
    jint computeDeviceOrdinal, jint classCount, jint treeCount,
    jint numFeaturesPerNode, jint minObservationsLeafNode,
    jint minObservationsSplitNode, jdouble minWeightFractionLeafNode,
    jdouble minImpurityDecreaseSplitNode, jboolean bootstrap,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    homogen_table hFeaturetable =
        *reinterpret_cast<const homogen_table *>(pNumTabFeature);
    homogen_table hLabeltable =
        *reinterpret_cast<const homogen_table *>(pNumTabLabel);
    std::cout << "doRFClassifierOneAPICompute get_column_count = "
              << hFeaturetable.get_column_count() << std::endl;
    std::cout << "doRFClassifierOneAPICompute classCount = " << classCount
              << std::endl;
    const auto df_desc =
        df::descriptor<float, df::method::hist, df::task::classification>{}
            .set_class_count(classCount)
            .set_tree_count(treeCount)
            .set_features_per_node(numFeaturesPerNode)
            .set_min_observations_in_leaf_node(minObservationsLeafNode)
            .set_min_observations_in_split_node(minObservationsSplitNode)
            .set_min_weight_fraction_in_leaf_node(minWeightFractionLeafNode)
            .set_min_impurity_decrease_in_split_node(
                minImpurityDecreaseSplitNode)
            .set_error_metric_mode(df::error_metric_mode::out_of_bag_error)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_infer_mode(df::infer_mode::class_responses |
                            df::infer_mode::class_probabilities)
            .set_voting_mode(df::voting_mode::weighted);

    const auto result_train =
        preview::train(comm, df_desc, hFeaturetable, hLabeltable);
    const auto result_infer =
        preview::infer(comm, df_desc, result_train.get_model(), hFeaturetable);
    jobject trees = nullptr;
    if (comm.get_rank() == 0) {
        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;
        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
        std::cout << "Prediction results:\n"
                  << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n"
                  << result_infer.get_probabilities() << std::endl;

        // convert c++ map to java hashmap
        collect_model(env, result_train.get_model(), classCount, treeForest);
        trees = convertJavaMap(env, treeForest, classCount);

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID predictionNumericTableField =
            env->GetFieldID(clazz, "predictionNumericTable", "J");
        jfieldID probabilitiesNumericTableField =
            env->GetFieldID(clazz, "probabilitiesNumericTable", "J");
        jfieldID importancesNumericTableField =
            env->GetFieldID(clazz, "importancesNumericTable", "J");
        HomogenTablePtr prediction =
            std::make_shared<homogen_table>(result_infer.get_responses());
        saveHomogenTablePtrToVector(prediction);

        HomogenTablePtr probabilities =
            std::make_shared<homogen_table>(result_infer.get_probabilities());
        saveHomogenTablePtrToVector(probabilities);

        HomogenTablePtr importances =
            std::make_shared<homogen_table>(result_train.get_var_importance());
        saveHomogenTablePtrToVector(importances);

        // Set prediction for result
        env->SetLongField(resultObj, predictionNumericTableField,
                          (jlong)prediction.get());

        // Set probabilities for result
        env->SetLongField(resultObj, probabilitiesNumericTableField,
                          (jlong)probabilities.get());

       // Set importances for result
       env->SetLongField(resultObj, importancesNumericTableField,
                         (jlong)importances.get());
    }
    return trees;
}

/*
 * Class:     com_intel_oap_mllib_classification_RandomForestClassifierDALImpl
 * Method:    cRFClassifierTrainDAL
 * Signature:
 * (JJIIIIIIIDDZLjava/lang/String;Lcom/intel/oap/mllib/classification/RandomForestResult;)V
 */
JNIEXPORT jobject JNICALL
Java_com_intel_oap_mllib_classification_RandomForestClassifierDALImpl_cRFClassifierTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabFeature, jlong pNumTabLabel,
    jint executorNum, jint computeDeviceOrdinal, jint classCount,
    jint treeCount, jint numFeaturesPerNode, jint minObservationsLeafNode,
    jint minObservationsSplitNode, jdouble minWeightFractionLeafNode,
    jdouble minImpurityDecreaseSplitNode, jboolean bootstrap,
    jintArray gpuIdxArray, jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
    ccl::communicator &cclComm = getComm();
    int rankId = cclComm.rank();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
    case ComputeDevice::gpu: {
        int nGpu = env->GetArrayLength(gpuIdxArray);
        std::cout << "oneDAL (native): use GPU kernels with " << nGpu
                  << " GPU(s)"
                  << " rankid " << rankId << std::endl;

        jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);

        int size = cclComm.size();
        ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);

        auto queue =
            getAssignedGPU(device, cclComm, size, rankId, gpuIndices, nGpu);

        ccl::shared_ptr_class<ccl::kvs> &kvs = getKvs();
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                queue, size, rankId, kvs);
        jobject hashmapObj = doRFClassifierOneAPICompute(
            env, pNumTabFeature, pNumTabLabel, executorNum,
            computeDeviceOrdinal, classCount, treeCount, numFeaturesPerNode,
            minObservationsLeafNode, minObservationsSplitNode,
            minWeightFractionLeafNode, minImpurityDecreaseSplitNode, bootstrap,
            comm, resultObj);
        return hashmapObj;
    }
    }
    return nullptr;
}
#endif
