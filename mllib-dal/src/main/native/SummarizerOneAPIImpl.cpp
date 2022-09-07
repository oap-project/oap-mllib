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

using namespace std;
using namespace oneapi::dal;
const int ccl_root = 0;

static void doSummarizerOneAPICompute(JNIEnv *env, jint rankId,
                                      jlong pNumTabData, jint executorNum,
                                      const ccl::string &ipPort,
                                      jint computeDeviceOrdinal,
                                      jobject resultObj) {
    std::cout << "oneDAL (native): compute start , rankid = " << rankId
              << "; device = " << ComputeDeviceString[computeDeviceOrdinal]
              << std::endl;
    const bool isRoot = (rankId == ccl_root);
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto bs_desc = basic_statistics::descriptor{};
    auto queue = getQueue(device);
    auto comm = preview::spmd::make_communicator<preview::spmd::backend::ccl>(
        queue, executorNum, rankId, ipPort);
    const auto result_train = preview::compute(comm, bs_desc, htable);

    if (isRoot) {
        std::cout << "Minimum:\n" << result_train.get_min() << std::endl;
        std::cout << "Maximum:\n" << result_train.get_max() << std::endl;
        std::cout << "Mean:\n" << result_train.get_mean() << std::endl;
        std::cout << "Variance:\n" << result_train.get_variance() << std::endl;

        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID meanTableField = env->GetFieldID(clazz, "meanNumericTable", "J");
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
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executorNum,
    jint computeDeviceOrdinal, jint rankId, jstring ipPort, jobject resultObj) {
    const char *ipPortPtr = env->GetStringUTFChars(ipPort, 0);
    std::string ipPortStr = std::string(ipPortPtr);
    doSummarizerOneAPICompute(env, rankId, pNumTabData, executorNum, ipPortStr,
                              computeDeviceOrdinal, resultObj);
    env->ReleaseStringUTFChars(ipPort, ipPortPtr);
    return 0;
}
#endif
