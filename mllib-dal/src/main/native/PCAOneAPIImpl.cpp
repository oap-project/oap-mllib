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
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_feature_PCADALImpl.h"
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;


static void doPCAOneAPICompute(JNIEnv *env, jint rankId, jlong pNumTabData,
                               jint executorNum, const ccl::string &ipPort,
                               jint computeDeviceOrdinal, jobject resultObj) {
    std::cout << "oneDAL (native): GPU/CPU compute start , rankid = " << rankId
                  << "; device = " << computeDeviceOrdinal << "(0:HOST;1:CPU;2:GPU)"
                  << std::endl;
    const bool isRoot = (rankId == ccl_root);
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto pca_desc = pca::descriptor{};
    auto queue = getQueue(device);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executorNum, rankId, ipPort);

    pca::train_input local_input{htable};
    const auto result_train = preview::train(comm, pca_desc, local_input);
    if (isRoot) {
        // Return all eigenvalues & eigenvectors
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID pcNumericTableField =
            env->GetFieldID(clazz, "pcNumericTable", "J");
        jfieldID explainedVarianceNumericTableField =
            env->GetFieldID(clazz, "explainedVarianceNumericTable", "J");

        HomogenTablePtr eigenvectors =
            std::make_shared<homogen_table>(result_train.get_eigenvectors());
        saveHomogenTablePtrToVector(eigenvectors);

        HomogenTablePtr eigenvalues =
            std::make_shared<homogen_table>(result_train.get_eigenvalues());
        saveHomogenTablePtrToVector(eigenvalues);

        env->SetLongField(resultObj, pcNumericTableField,
                          (jlong)eigenvectors.get());
        env->SetLongField(resultObj, explainedVarianceNumericTableField,
                          (jlong)eigenvalues.get());
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_feature_PCADALImpl_cPCATrainDAL(
<<<<<<< HEAD
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executorNum,
    jint computeDeviceOrdinal, jint rankId, jstring ipPort, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels " << std::endl;
    const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
    std::string ipPortStr = std::string(ipPortPtr);
    doPCAOneAPICompute(env, rankId, pNumTabData, executorNum, ipPortStr,
                       computeDeviceOrdinal, resultObj);
    env->ReleaseStringUTFChars(ipPort, ipPortPtr);
=======
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executor_num,
    jint cComputeDevice, jint rankId, jstring ip_port, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    const char *ipport = env->GetStringUTFChars(ip_port, 0);
    std::string ipPort = std::string(ipport);
    printf("oneDAL (native):  PCATrainDAL %d \n", cComputeDevice);
    compute_device device = getComputeDevice(cComputeDevice);
    switch (device) {
    case compute_device::host: {
        printf("oneDAL (native):  PCATrainDAL host \n");
        doPCAHOSTOneAPICompute(env, rankId, pNumTabData, executor_num, ipPort,
                               resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case compute_device::cpu:
    case compute_device::gpu: {
        cout << "oneDAL (native): use DPCPP GPU/CPU kernels" << endl;
        auto queue = getQueue(device);
        doPCACPUorGPUOneAPICompute(env, rankId, pNumTabData, executor_num,
                                   ipPort, queue, resultObj);
        break;
    }
#endif
    }
    env->ReleaseStringUTFChars(ip_port, ipport);
>>>>>>> update
    return 0;
}
#endif
