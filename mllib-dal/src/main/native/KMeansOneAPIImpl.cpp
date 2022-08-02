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
#include <mutex>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

static jlong doKMeansOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabData,
                                   jlong pNumTabCenters, jint clusterNum,
                                   jdouble tolerance, jint iterationNum,
                                   jint executorNum, const ccl::string &ipPort,
                                   jint computeDeviceOrdinal,
                                   jobject resultObj) {
    std::cout << "oneDAL (native): GPU/CPU compute start , rankid = " << rankId
              << "; device = " << computeDeviceOrdinal << "(0:HOST;1:GPU;2:CPU)"
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    compute_device device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);
    homogen_table centroids =
        *reinterpret_cast<const homogen_table *>(pNumTabCenters);
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(clusterNum)
                                 .set_max_iteration_count(iterationNum)
                                 .set_accuracy_threshold(tolerance);
    kmeans::train_input local_input{htable, centroids};
    auto queue = getQueue(device);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executorNum, rankId, ipPort);
    kmeans::train_result result_train =
        preview::train(comm, kmeans_desc, local_input);
    if (isRoot) {
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");
        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField,
                         result_train.get_iteration_count());
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField,
                            result_train.get_objective_function_value());

        HomogenTablePtr centroidsPtr = std::make_shared<homogen_table>(
            result_train.get_model().get_centroids());
        saveHomogenTablePtrToVector(centroidsPtr);
        return (jlong)centroidsPtr.get();
    } else {
        return (jlong)0;
    }
}

/*
 * Class:     com_intel_oap_mllib_clustering_KMeansDALImpl
 * Method:    cKMeansOneapiComputeWithInitCenters
 * Signature: (JJIDIIILcom/intel/oap/mllib/clustering/KMeansResult;)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansOneapiComputeWithInitCenters(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters,
    jint clusterNum, jdouble tolerance, jint iterationNum, jint executorNum,
    jint computeDeviceOrdinal, jint rankId, jstring ipPort, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels " << std::endl;
    const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
    std::string ipPortStr = std::string(ipPortPtr);
    jlong ret = 0L;
    ret = doKMeansOneAPICompute(
        env, rankId, pNumTabData, pNumTabCenters, clusterNum, tolerance,
        iterationNum, executorNum, ipPortStr, computeDeviceOrdinal, resultObj);
    env->ReleaseStringUTFChars(ipPort, ipPortPtr);
    return ret;
}
#endif
