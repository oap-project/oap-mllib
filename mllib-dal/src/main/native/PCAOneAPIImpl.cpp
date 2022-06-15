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
#include <iostream>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
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

static void doPCAHOSTOneAPICompute(JNIEnv *env, jint rankId, jint k,
                                   jlong pNumTabData, jint executor_num,
                                   const ccl::string &ipPort,
                                   jobject resultObj) {
    std::cout << "oneDAL (native): HOST compute start , rankid %ld " << rankId
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto pca_desc =
        pca::descriptor{}.set_component_count(k).set_deterministic(true);

    auto result_train = train(pca_desc, htable);
    if (isRoot) {
        std::cout << "Eigenvectors:\n"
                  << result_train.get_eigenvectors() << std::endl;
        std::cout << "Eigenvalues:\n"
                  << result_train.get_eigenvalues() << std::endl;
        // Return all eigenvalues & eigenvectors
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID pcNumericTableField =
            env->GetFieldID(clazz, "pcNumericTable", "J");
        jfieldID explainedVarianceNumericTableField =
            env->GetFieldID(clazz, "explainedVarianceNumericTable", "J");

        homogenPtr eigenvectors =
            std::make_shared<homogen_table>(result_train.get_eigenvectors());
        saveShareHomogenPtrVector(eigenvectors);

        homogenPtr eigenvalues =
            std::make_shared<homogen_table>(result_train.get_eigenvalues());
        saveShareHomogenPtrVector(eigenvalues);
        printf("eigenvalues.get() %ld \n ", eigenvalues.get());
        env->SetLongField(resultObj, pcNumericTableField,
                          (jlong)eigenvectors.get());
        env->SetLongField(resultObj, explainedVarianceNumericTableField,
                          (jlong)eigenvalues.get());
    }
}

static void doPCACPUorGPUOneAPICompute(JNIEnv *env, jint rankId, jint k,
                                       jlong pNumTabData, jint executor_num,
                                       const ccl::string &ipPort,
                                       cl::sycl::queue &queue,
                                       jobject resultObj) {
    std::cout << "oneDAL (native): GPU/CPU compute start , rankid %ld "
              << rankId << std::endl;
    const bool isRoot = (rankId == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto pca_desc =
        pca::descriptor{}.set_component_count(k).set_deterministic(true);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executor_num, rankId, ipPort);
    auto result_train = preview::train(comm, pca_desc, htable);
    if (isRoot) {
        std::cout << "Eigenvectors:\n"
                  << result_train.get_eigenvectors() << std::endl;
        std::cout << "Eigenvalues:\n"
                  << result_train.get_eigenvalues() << std::endl;
        // Return all eigenvalues & eigenvectors
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID pcNumericTableField =
            env->GetFieldID(clazz, "pcNumericTable", "J");
        jfieldID explainedVarianceNumericTableField =
            env->GetFieldID(clazz, "explainedVarianceNumericTable", "J");

        homogenPtr eigenvectors =
            std::make_shared<homogen_table>(result_train.get_eigenvectors());
        saveShareHomogenPtrVector(eigenvectors);

        homogenPtr eigenvalues =
            std::make_shared<homogen_table>(result_train.get_eigenvalues());
        saveShareHomogenPtrVector(eigenvalues);

        env->SetLongField(resultObj, pcNumericTableField,
                          (jlong)eigenvectors.get());
        env->SetLongField(resultObj, explainedVarianceNumericTableField,
                          (jlong)eigenvalues.get());
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_feature_PCADALImpl_cPCATrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint k, jint executor_num,
    jint cComputeDevice, jint rankId, jstring ip_port, jobject resultObj) {
    std::cout << "oneDAL (native): use GPU DPC++ kernels with " << std::endl;
    const char *ipport = env->GetStringUTFChars(ip_port, 0);
    std::string ipPort = std::string(ipport);
    printf("oneDAL (native):  PCATrainDAL %d \n", cComputeDevice);
    compute_device device = getComputeDevice(cComputeDevice);
    switch (device) {
    case compute_device::host: {
        printf("oneDAL (native):  PCATrainDAL host \n");
        doPCAHOSTOneAPICompute(env, rankId, k, pNumTabData, executor_num,
                               ipPort, resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case compute_device::cpu:
    case compute_device::gpu: {
        cout << "oneDAL (native): use DPCPP GPU/CPU kernels" << endl;
        auto queue = getQueue(device);
        doPCACPUorGPUOneAPICompute(env, rankId, k, pNumTabData, executor_num,
                                   ipPort, queue, resultObj);
        break;
    }
#endif
    }
    env->ReleaseStringUTFChars(ip_port, ipport);
    return 0;
}
