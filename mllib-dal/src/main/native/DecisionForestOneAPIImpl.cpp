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
#include <iterator>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_classification_RandomForestClassifierDALImpl.h"
#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
namespace df = oneapi::dal::decision_forest;
const int ccl_root = 0;
//const char* LEAFNODE_CLASS_NAME = "org.apache.spark.ml.tree.LeafNode";
//const char* INTERNALNODE_CLASS_NAME = "org.apache.spark.ml.tree.InternalNode";
//const char* SPLIT_CLASS_NAME = "org/apache/spark/ml/tree/Split";
//const char* IMPURITY_CALCULATOR_CLASS_NAME = "org.apache.spark.mllib.tree.impurity.ImpurityCalculator";
std::vector<jobject> g_ObjVector;

void saveObjToVector(const jobject &ptr) {
    g_ObjVector.push_back(ptr);
}

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
const std::shared_ptr<std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>> treeForest =
        std::make_shared<std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>>();

LearningNode convertsplitToLearningNode(const df::split_node_info<df::task::classification>& info, const int classCount){
           LearningNode splitNode;
           splitNode.isLeaf = false;
           splitNode.level = info.get_level();
           splitNode.splitIndex = info.get_feature_index();
           splitNode.splitValue = info.get_feature_value();
           splitNode.impurity = info.get_impurity();
           splitNode.sampleCount = info.get_sample_count();
           return splitNode;
}

LearningNode convertleafToLearningNode(const df::leaf_node_info<df::task::classification>& info, const int classCount){
           LearningNode leafNode;
           leafNode.isLeaf = true;
           leafNode.level = info.get_level();
           leafNode.impurity = info.get_impurity();
           leafNode.sampleCount = info.get_sample_count();
           std::unique_ptr<double[]> arr(new double[classCount]);
           for (std::int64_t index_class = 0; index_class < classCount; ++index_class) {
                arr[index_class] = info.get_probability(index_class);
           }
           leafNode.probability = std::move(arr);
           return leafNode;
}

struct collect_nodes{
    std::shared_ptr<std::vector<LearningNode>> treesVector;
    jint classCount;
    collect_nodes(const std::shared_ptr<std::vector<LearningNode>> &vector, int count) {
        treesVector = vector;
        classCount= count;
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

        treesVector->push_back(convertleafToLearningNode(info, classCount));
        std::cout << str << std::endl;

        return true;
        }

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

        treesVector->push_back(convertsplitToLearningNode(info, classCount));

        std::cout << str << std::endl;
        return true;
    }
};

template <typename Task>
void collect_model(JNIEnv *env, const df::model<Task>& m,
                                const jint classCount,
                                const std::shared_ptr<std::map<std::int64_t,std::shared_ptr<std::vector<LearningNode>>>> &treeForest) {

    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;
        std::shared_ptr<std::vector<LearningNode>> myVec = std::make_shared<std::vector<LearningNode>>();
        m.traverse_depth_first(i, collect_nodes{myVec, classCount});
        treeForest->insert({i , myVec});
    }
}

jobject convertJavaMap(JNIEnv *env,
                       const std::shared_ptr<std::map<std::int64_t, std::shared_ptr<std::vector<LearningNode>>>> map,
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
    jclass learningNodeClass = env->FindClass("com/intel/oap/mllib/classification/LearningNode");
    jmethodID learningNodeConstructor = env->GetMethodID(learningNodeClass, "<init>", "()V");

    // Iterate over the C++ map and add each entry to the Java map
    for (const auto& entry : *map) {
        std::cout << "Iterate over the C++ map and add each entry to the Java map" << std::endl;
        // Create a new Java ArrayList to hold the LearningNode objects
        jobject jList = env->NewObject(listClass, listConstructor);

        // Iterate over the LearningNode objects and add each one to the ArrayList
        for (const auto& node : *entry.second) {
            jobject jNode = env->NewObject(learningNodeClass, learningNodeConstructor);

            // Set the fields of the Java LearningNode object using JNI functions
            jfieldID levelField = env->GetFieldID(learningNodeClass, "level", "I");
            env->SetIntField(jNode, levelField, node.level);

            jfieldID impurityField = env->GetFieldID(learningNodeClass, "impurity", "D");
            env->SetDoubleField(jNode, impurityField, node.impurity);

            jfieldID splitIndexField = env->GetFieldID(learningNodeClass, "splitIndex", "I");
            env->SetIntField(jNode, splitIndexField, node.splitIndex);

            jfieldID splitValueField = env->GetFieldID(learningNodeClass, "splitValue", "D");
            env->SetDoubleField(jNode, splitValueField, node.splitValue);

            jfieldID isLeafField = env->GetFieldID(learningNodeClass, "isLeaf", "Z");
            env->SetBooleanField(jNode, isLeafField, node.isLeaf);

            // Convert the probability array
            if (node.probability != nullptr) {
                std::cout << "convertJavaMap node.probability  = " <<  node.probability.get()[0]
                                       << std::endl;
                jfieldID probabilityField = env->GetFieldID(learningNodeClass, "probability", "[D");
                jdoubleArray jProbability = env->NewDoubleArray(classCount);
//                env->SetDoubleArrayRegion(jProbability, 0, classCount, node.probability.get());
                // call the native method to get the values
                jdouble* elements = env->GetDoubleArrayElements(jProbability, nullptr);
                // Do something with the array elements
                for (int i = 0; i < classCount; i++) {
                    elements[i] = node.probability.get()[i];
                }
                env->SetDoubleArrayRegion(jProbability, 0, classCount, elements);
                jdouble* data = env->GetDoubleArrayElements(jProbability, NULL);

                for (int i = 0; i < classCount; i++) {
                    std::cout << "convertJavaMap jProbability value : " << data[i] << std::endl;
                }
                env->SetObjectField(jNode, probabilityField, jProbability);
                jobject probability_object = env->GetObjectField(jNode, probabilityField);
                if (probability_object == NULL) {
                  std::cout << "probability_object null " << std::endl;
                  // An exception occurred
                  exit(-1);
                }
                jdoubleArray probability_array = reinterpret_cast<jdoubleArray>(probability_object);
                jdouble* probability_data = env->GetDoubleArrayElements(probability_array, NULL);
                if (probability_data == NULL) {
                   std::cout << "probability_data null " << std::endl;
                   // An exception occurred
                   exit(-1);
                }
                for (std::int64_t index_class = 0; index_class < classCount; ++index_class) {
                   std::cout << "convertleafToLearningNode get probability : " << probability_data[index_class] << std::endl;
                }
            }

            jfieldID probability_Field = env->GetFieldID(learningNodeClass, "probability", "[D");
            jobject probabilityObject = env->GetObjectField(jNode, probability_Field);
            if (probabilityObject == NULL) {
              std::cout << "probabilityObject null " << std::endl;
              // An exception occurred
              exit(-1);
            }
            jdoubleArray probabilityArray = reinterpret_cast<jdoubleArray>(probabilityObject);
            jdouble* probabilityData = env->GetDoubleArrayElements(probabilityArray, NULL);
            if (probabilityData == NULL) {
               std::cout << "probabilityData null " << std::endl;
               // An exception occurred
               exit(-1);
            }
            for (std::int64_t index_class = 0; index_class < classCount; ++index_class) {
               std::cout << "convertleafToLearningNode get probability data : " << probabilityData[index_class] << std::endl;
            }

            jfieldID sampleCountField = env->GetFieldID(learningNodeClass, "sampleCount", "I");
            env->SetIntField(jNode, sampleCountField, node.sampleCount);

            // Add the LearningNode object to the ArrayList
            jmethodID listAdd = env->GetMethodID(listClass, "add", "(Ljava/lang/Object;)Z");
            env->CallBooleanMethod(jList, listAdd, jNode);
        }

        // Add the ArrayList to the Java map
        jmethodID mapPut = env->GetMethodID(mapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
        std::cout << "convertJavaMap tree id  = " <<  entry.first
                           << std::endl;
        // Create a new Integer object with the value key
        jobject jKey = env->NewObject(
          env->FindClass("java/lang/Integer"), // Find the Integer class
          env->GetMethodID(env->FindClass("java/lang/Integer"), "<init>", "(I)V"), // Get the constructor method
          (jint) static_cast<jint>(entry.first)
        );
        env->CallObjectMethod(jMap, mapPut, jKey, jList);

    }
    std::cout << "convert map to HashMap end " << std::endl;

    return jMap;
}

static jobject doRFClassifierOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabFeature,
                                           jlong pNumTabLabel, jint executorNum,
                                           jint computeDeviceOrdinal, jint classCount, jint treeCount,
                                           jint minObservationsLeafNode, jint minObservationsSplitNode,
                                           jdouble minWeightFractionLeafNode, jdouble minImpurityDecreaseSplitNode,
                                           jboolean bootstrap,
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
  std::cout << "doRFClassifierOneAPICompute get_column_count = " <<  hFeaturetable.get_column_count()
                   << std::endl;
  std::cout << "doRFClassifierOneAPICompute classCount = " <<  classCount
                   << std::endl;
   const auto df_desc =
        df::descriptor<float, df::method::hist, df::task::classification>{}
            .set_class_count(classCount)
            .set_tree_count(treeCount)
            .set_features_per_node(hFeaturetable.get_column_count())
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
        preview::train(comm, df_desc, hFeaturetable, hLabeltable);
    const auto result_infer =
        preview::infer(comm, df_desc, result_train.get_model(), hFeaturetable);
    jobject trees = nullptr;
    if (comm.get_rank() == 0) {
        std::cout << "Variable importance results:\n"
                  << result_train.get_var_importance() << std::endl;
        std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;
        std::cout << "Probabilities results:\n" << result_infer.get_probabilities() << std::endl;

        // convert c++ map to java hashmap
        collect_model(env, result_train.get_model(), classCount, treeForest);
        trees = convertJavaMap(env, treeForest, classCount);

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID predictionNumericTableField = env->GetFieldID(clazz, "predictionNumericTable", "J");
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

        saveObjToVector(trees);
       }
       return trees;
}

/*
 * Class:     com_intel_oap_mllib_classification_RandomForestClassifierDALImpl
 * Method:    cRFClassifierTrainDAL
 * Signature: (JJIIIIIIIDDZLjava/lang/String;Lcom/intel/oap/mllib/classification/RandomForestResult;)V
 */
JNIEXPORT jobject JNICALL Java_com_intel_oap_mllib_classification_RandomForestClassifierDALImpl_cRFClassifierTrainDAL
  (JNIEnv *env, jobject obj, jlong pNumTabFeature, jlong pNumTabLabel, jint executorNum,
  jint computeDeviceOrdinal, jint classCount, jint rankId, jint treeCount,
  jint minObservationsLeafNode, jint minObservationsSplitNode, jdouble minWeightFractionLeafNode,
  jdouble minImpurityDecreaseSplitNode, jboolean bootstrap, jstring ipPort,
  jobject resultObj) {
      std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
      const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
      std::string ipPortStr = std::string(ipPortPtr);
      jobject hashmapObj = doRFClassifierOneAPICompute(
          env, rankId, pNumTabFeature, pNumTabLabel, executorNum, computeDeviceOrdinal, classCount,
          treeCount, minObservationsLeafNode, minObservationsSplitNode,
          minWeightFractionLeafNode, minImpurityDecreaseSplitNode, bootstrap,
          ipPortPtr, resultObj);
      env->ReleaseStringUTFChars(ipPort, ipPortPtr);
      return hashmapObj;
  }
#endif
