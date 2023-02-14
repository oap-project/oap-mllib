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
const Map<int, vector<char>> treeForest;
vector<char> vector;

struct trees_wrap{

    bool operator()(const df::leaf_node_info<df::task::classification>& info) {
        std::string str;
        str.append(to_string(info.get_level()));
        str.append("|" );
        str.append("leaf");
        str.append("|" );
        str.append(to_string(info.get_response()));
        str.append("|" );
        str.append(to_string(info.get_impurity()));
        str.append("|" );
        str.append(to_string(info.get_sample_count()));

        vector.push_back(str);
        std::cout << str << std::endl;

        return true;

    bool operator()(const df::split_node_info<df::task::classification>& info) {
        std::string str;
        str.append(to_string(info.get_level()));
        str.append("|" );
        str.append("split");
        str.append("|" );
        str.append(to_string(info.get_feature_index()));
        str.append("|" );
        str.append(to_string(info.get_feature_value()));
        str.append("|" );
        str.append(to_string(info.get_impurity()));
        str.append("|" );
        str.append(to_string(info.get_sample_count()));
        str.append(to_string(info.get_sample_count()));

        vector.push_back(str);

        std::cout << str << std::endl;
        return true;
    }
};

template <typename Task>
void print_model(const df::model<Task>& m) {
    std::cout << "Number of trees: " << m.get_tree_count() << std::endl;
    for (std::int64_t i = 0, n = m.get_tree_count(); i < n; ++i) {
        std::cout << "Tree #" << i << std::endl;
        m.traverse_depth_first(i, trees_wrap{});
        treeForest.insert({ i , vector})
    }
}

static jlong doRFClassifierOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabFeature,
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

    }

}

/*
 * Class:     com_intel_oap_mllib_classification_RandomForestClassifierDALImpl
 * Method:    cRFClassifierTrainDAL
 * Signature: (JIIIIIIIIDDLjava/lang/String;Lcom/intel/oap/mllib/classification/RandomForestResult;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oap_mllib_classification_RandomForestClassifierDALImpl_cRFClassifierTrainDAL
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
