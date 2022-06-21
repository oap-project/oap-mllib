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

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

static void doCorrelationHOSTOneAPICompute(JNIEnv *env, jint rankId,
                                           jlong pNumTabData, jint executor_num,
                                           const ccl::string &ipPort,
                                           jobject resultObj) {
    std::cout << "oneDAL (native): HOST compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto cor_desc = covariance::descriptor{}.set_result_options(
        covariance::result_options::cor_matrix |
        covariance::result_options::means);
    const auto result_train = compute(cor_desc, htable);
    if (isRoot) {
        std::cout << "Mean:\n" << result_train.get_means() << std::endl;
        std::cout << "Correlation:\n"
                  << result_train.get_cor_matrix() << std::endl;
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        homogenPtr correlation =
            std::make_shared<homogen_table>(result_train.get_cor_matrix());
        saveShareHomogenPtrVector(correlation);

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation.get());
    }
}

static void
doCorrelationCPUorGPUOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabData,
                                   jint executor_num, const ccl::string &ipPort,
                                   cl::sycl::queue &queue, jobject resultObj) {
    std::cout << "oneDAL (native): GPU/CPU compute start , rankid %ld "
              << rankId << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto cor_desc = covariance::descriptor{}.set_result_options(
        covariance::result_options::cor_matrix |
        covariance::result_options::means);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executor_num, rankId, ipPort);
    const auto result_train = preview::compute(comm, cor_desc, htable);
    if (isRoot) {
        std::cout << "Mean:\n" << result_train.get_means() << std::endl;
        std::cout << "Correlation:\n"
                  << result_train.get_cor_matrix() << std::endl;
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        homogenPtr correlation =
            std::make_shared<homogen_table>(result_train.get_cor_matrix());
        saveShareHomogenPtrVector(correlation);

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation.get());
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executor_num,
    jint cComputeDevice, jint rankId, jstring ip_port, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    const char *ipport = env->GetStringUTFChars(ip_port, 0);
    std::string ipPort = std::string(ipport);
    printf("oneDAL (native):  CorrelationTrainDAL %d \n", cComputeDevice);
    compute_device device = getComputeDevice(cComputeDevice);
    switch (device) {
    case compute_device::host: {
        printf("oneDAL (native):  CorrelationTrainDAL host \n");
        doCorrelationHOSTOneAPICompute(env, rankId, pNumTabData, executor_num,
                                       ipPort, resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case compute_device::cpu:
    case compute_device::gpu: {
        cout << "oneDAL (native): use DPCPP GPU/CPU kernels" << endl;
        auto queue = getQueue(device);
        doCorrelationCPUorGPUOneAPICompute(
            env, rankId, pNumTabData, executor_num, ipPort, queue, resultObj);
        break;
    }
#endif
    }
    env->ReleaseStringUTFChars(ip_port, ipport);
    return 0;
}
