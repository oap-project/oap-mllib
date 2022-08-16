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
#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

typedef std::shared_ptr<homogen_table> homogenPtr;

std::mutex kmtx;
std::vector<homogenPtr> cVector;

static void saveShareHomogenPtrVector(const homogenPtr &ptr) {
    kmtx.lock();
    cVector.push_back(ptr);
    kmtx.unlock();
}

static jlong doKMeansOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabData,
                                   jlong pNumTabCenters, jint cluster_num,
                                   jdouble tolerance, jint iteration_num,
                                   jint executor_num, const ccl::string &ipPort,
                                   jint cComputeDevice, jobject resultObj) {
    std::cout << "oneDAL (native): OneAPI compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    compute_device device = getComputeDevice(cComputeDevice);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);
    homogen_table centroids =
        *reinterpret_cast<const homogen_table *>(pNumTabCenters);
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(cluster_num)
                                 .set_max_iteration_count(iteration_num)
                                 .set_accuracy_threshold(tolerance);
    kmeans::train_input local_input{htable, centroids};
    auto queue = getQueue(device);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executor_num, rankId, ipPort);
    auto t1 = std::chrono::high_resolution_clock::now();
    kmeans::train_result result_train =
            preview::train(comm, kmeans_desc, local_input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "KMeans (native) RankId = << " << rankId
              << "; spend training times : " << duration
                  << " secs" << std::endl;
    if (isRoot) {
        t2 = std::chrono::high_resolution_clock::now();
        duration =
                std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        std::cout << "KMeans (native) spend training times : " << duration
                      << " secs" << std::endl;
        std::cout << "iteration_num: " << iteration_num << std::endl;
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                  << std::endl;
        std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
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

        homogenPtr centroidsPtr = std::make_shared<homogen_table>(
            result_train.get_model().get_centroids());
        saveShareHomogenPtrVector(centroidsPtr);
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
    jint cluster_num, jdouble tolerance, jint iteration_num, jint executor_num,
    jint cComputeDevice, jint rankId, jstring ip_port, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    const char *ipport = env->GetStringUTFChars(ip_port, 0);
    std::string ipPort = std::string(ipport);
    jlong ret = 0L;
    printf("oneDAL (native):  KMeansOneapiComputeWithInitCenters %d \n",
           cComputeDevice);
    ret = doKMeansOneAPICompute(
        env, rankId, pNumTabData, pNumTabCenters, cluster_num, tolerance,
        iteration_num, executor_num, ipPort, cComputeDevice, resultObj);
    env->ReleaseStringUTFChars(ip_port, ipport);
    return ret;
}
#endif
