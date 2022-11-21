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
#include "com_intel_oap_mllib_stat_SummarizerDALImpl.h"
#include "oneapi/dal/algo/basic_statistics.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "service.h"
#include "OneCCL.h"

using namespace std;
using namespace oneapi::dal;

static void doSummarizerOneAPICompute(JNIEnv *env,
                                      jlong pNumTabData,
                                      preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
                                      jobject resultObj) {
    std::cout << "oneDAL (native): compute start " << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto bs_desc = basic_statistics::descriptor{};
    const auto result_train = preview::compute(comm, bs_desc, htable);
    if (isRoot) {
        std::cout << "Minimum:\n" << result_train.get_min() << std::endl;
        std::cout << "Maximum:\n" << result_train.get_max() << std::endl;
        std::cout << "Mean:\n" << result_train.get_mean() << std::endl;
        std::cout << "Variance:\n" << result_train.get_variance() << std::endl;

        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID meanTableField =
            env->GetFieldID(clazz, "meanNumericTable", "J");
        jfieldID varianceTableField =
            env->GetFieldID(clazz, "varianceNumericTable", "J");
        jfieldID minimumTableField =
            env->GetFieldID(clazz, "minimumNumericTable", "J");
        jfieldID maximumTableField =
            env->GetFieldID(clazz, "maximumNumericTable", "J");

        HomogenTablePtr meanTable =
            std::make_shared<homogen_table>(result_train.get_mean());
        saveHomogenTablePtrToVector(meanTable);
        HomogenTablePtr varianceTable =
            std::make_shared<homogen_table>(result_train.get_variance());
        saveHomogenTablePtrToVector(varianceTable);
        HomogenTablePtr maxTable =
            std::make_shared<homogen_table>(result_train.get_max());
        saveHomogenTablePtrToVector(maxTable);
        HomogenTablePtr minTable =
            std::make_shared<homogen_table>(result_train.get_min());
        saveHomogenTablePtrToVector(minTable);
        env->SetLongField(resultObj, meanTableField, (jlong)meanTable.get());
        env->SetLongField(resultObj, varianceTableField,
                          (jlong)varianceTable.get());
        env->SetLongField(resultObj, maximumTableField, (jlong)maxTable.get());
        env->SetLongField(resultObj, minimumTableField, (jlong)minTable.get());
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_SummarizerDALImpl_cSummarizerTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData,
    jint computeDeviceOrdinal, jintArray gpuIdxArray, jobject resultObj) {
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

    ccl::shared_ptr_class<ccl::kvs> &kvs  = getKvs();
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, size, rankId, kvs);
    doSummarizerOneAPICompute(env, pNumTabData, comm, resultObj);
    env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);

    return 0;
}
#endif
