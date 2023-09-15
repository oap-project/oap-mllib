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

void generateSplitLearningNode(
    JNIEnv *env, const df::split_node_info<df::task::classification> &info,
    const int classCount, jobject jList, jclass listClass,
    jclass learningNodeClass, jmethodID learningNodeConstructor) {
    jobject jNode = env->NewObject(learningNodeClass, learningNodeConstructor);

    // Set the fields of the Java LearningNode object using JNI
    // functions
    jfieldID levelField = env->GetFieldID(learningNodeClass, "level", "I");
    env->SetIntField(jNode, levelField, info.get_level());

    jfieldID impurityField =
        env->GetFieldID(learningNodeClass, "impurity", "D");
    env->SetDoubleField(jNode, impurityField, info.get_impurity());

    jfieldID splitIndexField =
        env->GetFieldID(learningNodeClass, "splitIndex", "I");
    env->SetIntField(jNode, splitIndexField, info.get_feature_index());

    jfieldID splitValueField =
        env->GetFieldID(learningNodeClass, "splitValue", "D");
    env->SetDoubleField(jNode, splitValueField, info.get_feature_value());

    jfieldID isLeafField = env->GetFieldID(learningNodeClass, "isLeaf", "Z");
    env->SetBooleanField(jNode, isLeafField, false);

    // Convert the probability array
    jfieldID probabilityField =
        env->GetFieldID(learningNodeClass, "probability", "[D");
    jdoubleArray jProbability = env->NewDoubleArray(classCount);
    // call the native method to get the values
    jdouble *elements = env->GetDoubleArrayElements(jProbability, nullptr);
    // Do something with the array elements
    for (int i = 0; i < classCount; i++) {
        elements[i] = 0.0;
    }
    env->SetDoubleArrayRegion(jProbability, 0, classCount, elements);
    env->SetObjectField(jNode, probabilityField, jProbability);

    jfieldID sampleCountField =
        env->GetFieldID(learningNodeClass, "sampleCount", "I");
    env->SetIntField(jNode, sampleCountField, info.get_sample_count());

    // Add the LearningNode object to the ArrayList
    jmethodID listAdd =
        env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
    env->CallBooleanMethod(jList, listAdd, jNode);
}

void generateLeafLearningNode(
    JNIEnv *env, const df::leaf_node_info<df::task::classification> &info,
    const int classCount, jobject jList, jclass listClass,
    jclass learningNodeClass, jmethodID learningNodeConstructor) {
    jobject jNode = env->NewObject(learningNodeClass, learningNodeConstructor);

    // Set the fields of the Java LearningNode object using JNI
    // functions
    jfieldID levelField = env->GetFieldID(learningNodeClass, "level", "I");
    env->SetIntField(jNode, levelField, info.get_level());

    jfieldID impurityField =
        env->GetFieldID(learningNodeClass, "impurity", "D");
    env->SetDoubleField(jNode, impurityField, info.get_impurity());

    jfieldID splitIndexField =
        env->GetFieldID(learningNodeClass, "splitIndex", "I");
    env->SetIntField(jNode, splitIndexField, 0);

    jfieldID splitValueField =
        env->GetFieldID(learningNodeClass, "splitValue", "D");
    env->SetDoubleField(jNode, splitValueField, 0.0);

    jfieldID isLeafField = env->GetFieldID(learningNodeClass, "isLeaf", "Z");
    env->SetBooleanField(jNode, isLeafField, true);

    // Convert the probability array
    jfieldID probabilityField =
        env->GetFieldID(learningNodeClass, "probability", "[D");
    jdoubleArray jProbability = env->NewDoubleArray(classCount);
    // call the native method to get the values
    jdouble *elements = env->GetDoubleArrayElements(jProbability, nullptr);
    // Do something with the array elements
    for (int i = 0; i < classCount; i++) {
        elements[i] = info.get_probability(i);
    }
    env->SetDoubleArrayRegion(jProbability, 0, classCount, elements);
    env->SetObjectField(jNode, probabilityField, jProbability);

    jfieldID sampleCountField =
        env->GetFieldID(learningNodeClass, "sampleCount", "I");
    env->SetIntField(jNode, sampleCountField, info.get_sample_count());

    // Add the LearningNode object to the ArrayList
    jmethodID listAdd =
        env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
    env->CallBooleanMethod(jList, listAdd, jNode);
}

struct collect_nodes {
    JNIEnv *env;
    jint classCount;
    jobject jList;
    jclass listClass;
    jclass learningNodeClass;
    jmethodID learningNodeConstructor;
    collect_nodes(JNIEnv *env, int count, jobject jList, jclass listClass,
                  jclass learningNodeClass, jmethodID learningNodeConstructor) {
        this->env = env;
        this->classCount = count;
        this->jList = jList;
        this->listClass = listClass;
        this->learningNodeClass = learningNodeClass;
        this->learningNodeConstructor = learningNodeConstructor;
    }
    bool operator()(const df::leaf_node_info<df::task::classification> &info) {
        generateLeafLearningNode(env, info, classCount, jList, listClass,
                                 learningNodeClass, learningNodeConstructor);
        return true;
    }

    bool operator()(const df::split_node_info<df::task::classification> &info) {
        generateSplitLearningNode(env, info, classCount, jList, listClass,
                                  learningNodeClass, learningNodeConstructor);
        return true;
    }
};

template <typename Task>
jobject collect_model(JNIEnv *env, const df::model<Task> &m,
                      const jint classCount) {

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

    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout
            << "Iterate over the C++ map and add each entry to the Java map"
            << std::endl;
        // Create a new Java ArrayList to hold the LearningNode objects
        jobject jList = env->NewObject(listClass, listConstructor);
        m.traverse_depth_first(i, collect_nodes{env, classCount, jList,
                                                listClass, learningNodeClass,
                                                learningNodeConstructor});
        // Add the ArrayList to the Java map
        jmethodID mapPut = env->GetMethodID(
            mapClass, "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        std::cout << "convertJavaMap tree id  = " << i << std::endl;
        // Create a new Integer object with the value key
        jobject jKey = env->NewObject(
            env->FindClass("java/lang/Integer"), // Find the Integer class
            env->GetMethodID(env->FindClass("java/lang/Integer"), "<init>",
                             "(I)V"), // Get the constructor method
            (jint) static_cast<jint>(i));
        env->CallObjectMethod(jMap, mapPut, jKey, jList);
    }
    return jMap;
}

static jobject doRFClassifierOneAPICompute(
    JNIEnv *env, jlong pNumTabFeature, jlong pNumTabLabel, jint executorNum,
    jint computeDeviceOrdinal, jint classCount, jint treeCount,
    jint numFeaturesPerNode, jint minObservationsLeafNode,
    jint minObservationsSplitNode, jdouble minWeightFractionLeafNode,
    jdouble minImpurityDecreaseSplitNode, jint maxTreeDepth, jlong seed,
    jint maxBins, jboolean bootstrap,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table hFeaturetable =
        *reinterpret_cast<const homogen_table *>(pNumTabFeature);
    homogen_table hLabeltable =
        *reinterpret_cast<const homogen_table *>(pNumTabLabel);
    std::cout << "doRFClassifierOneAPICompute get_column_count = "
              << hFeaturetable.get_column_count() << std::endl;
    std::cout << "doRFClassifierOneAPICompute classCount = " << classCount
              << std::endl;
    const auto df_desc =
        df::descriptor<algorithmFPType, df::method::hist,
                       df::task::classification>{}
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
            .set_voting_mode(df::voting_mode::weighted)
            .set_max_tree_depth(maxTreeDepth)
            .set_max_bins(maxBins);

    const auto result_train =
        preview::train(comm, df_desc, hFeaturetable, hLabeltable);
    const auto result_infer =
        preview::infer(comm, df_desc, result_train.get_model(), hFeaturetable);
    jobject trees = nullptr;
    if (isRoot) {
        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;
        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
        std::cout << "Prediction results:\n"
                  << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n"
                  << result_infer.get_probabilities() << std::endl;

        // convert to java hashmap
        trees = collect_model(env, result_train.get_model(), classCount);

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID predictionNumericTableField =
            env->GetFieldID(clazz, "predictionNumericTable", "J");
        jfieldID probabilitiesNumericTableField =
            env->GetFieldID(clazz, "probabilitiesNumericTable", "J");
        HomogenTablePtr prediction =
            std::make_shared<homogen_table>(result_infer.get_responses());
        saveHomogenTablePtrToVector(prediction);

        HomogenTablePtr probabilities =
            std::make_shared<homogen_table>(result_infer.get_probabilities());
        saveHomogenTablePtrToVector(probabilities);

        // Set prediction for result
        env->SetLongField(resultObj, predictionNumericTableField,
                          (jlong)prediction.get());

        // Set probabilities for result
        env->SetLongField(resultObj, probabilitiesNumericTableField,
                          (jlong)probabilities.get());
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
    jdouble minImpurityDecreaseSplitNode, jint maxTreeDepth, jlong seed,
    jint maxBins, jboolean bootstrap, jintArray gpuIdxArray,
    jobject resultObj) {
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
            minWeightFractionLeafNode, minImpurityDecreaseSplitNode,
            maxTreeDepth, seed, maxBins, bootstrap, comm, resultObj);
        return hashmapObj;
    }
    default: {
        std::cout << "RandomForest (native): The compute device "
                  << "is not supported!" << std::endl;
        exit(-1);
    }
    }
    return nullptr;
}
#endif
