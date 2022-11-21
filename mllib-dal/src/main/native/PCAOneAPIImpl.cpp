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
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "Communicator.hpp"
#include "OneCCL.h"
#include "OutputHelpers.hpp"
#include "com_intel_oap_mllib_feature_PCADALImpl.h"
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

static void doPCAOneAPICompute(
    JNIEnv *env, jlong pNumTabData,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto pca_desc = pca::descriptor{};
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
    JNIEnv *env, jobject obj, jlong pNumTabData, jint computeDeviceOrdinal,
    jintArray gpuIdxArray, jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels " << std::endl;
    ccl::communicator &cclComm = getComm();
    int rankId = cclComm.rank();
    int nGpu = env->GetArrayLength(gpuIdxArray);
    std::cout << "oneDAL (native): use GPU kernels with " << nGpu << " GPU(s)"
              << std::endl;

    jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);

    int size = cclComm.size();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);

    auto queue =
        getAssignedGPU(device, cclComm, size, rankId, gpuIndices, nGpu);

    ccl::shared_ptr_class<ccl::kvs> &kvs = getKvs();
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, size, rankId, kvs);
    doPCAOneAPICompute(env, pNumTabData, comm, resultObj);
    env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);

    return 0;
}
#endif
