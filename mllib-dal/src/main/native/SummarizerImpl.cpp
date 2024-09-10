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
#include "Common.hpp"
#include "oneapi/dal/algo/basic_statistics.hpp"
#endif

#include "Logger.h"
#include "OneCCL.h"
#include "com_intel_oap_mllib_stat_SummarizerDALImpl.h"
#include "service.h"

using namespace std;
#ifdef CPU_GPU_PROFILE
using namespace oneapi::dal;
#endif
using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;

static void doSummarizerDAALCompute(JNIEnv *env, jobject obj, size_t rankId,
                                    ccl::communicator &comm,
                                    const NumericTablePtr &pData,
                                    size_t nBlocks, jobject resultObj) {
    logger::println(logger::INFO, "oneDAL (native): CPU compute start");
    using daal::byte;
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool isRoot = (rankId == ccl_root);

    low_order_moments::Distributed<step1Local, CpuAlgorithmFPType>
        localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(low_order_moments::data, pData);

    /* Compute low_order_moments */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger::println(logger::INFO,
                    "low_order_moments (native): local step took %d secs",
                    duration / 1000);

    t1 = std::chrono::high_resolution_clock::now();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    serializedData =
        services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);

    byte *nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(),
                                      perNodeArchLength); // 4 x "14016"

    /* Transfer partial results to step 2 on the root node */
    ccl::gather((int8_t *)nodeResults, perNodeArchLength,
                (int8_t *)(serializedData.get()), perNodeArchLength, comm)
        .wait();
    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger::println(logger::INFO,
                    "low_order_moments (native): ccl_gather took %d secs",
                    duration / 1000);
    if (isRoot) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /* Create an algorithm to compute covariance on the master node */
        low_order_moments::Distributed<step2Master, CpuAlgorithmFPType>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            low_order_moments::PartialResultPtr dataForStep2FromStep1(
                new low_order_moments::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(low_order_moments::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Set the parameter to choose the type of the output matrix */
        masterAlgorithm.parameter.estimatesToCompute =
            low_order_moments::estimatesAll;

        /* Merge and finalizeCompute covariance decomposition on the master node
         */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        low_order_moments::ResultPtr result = masterAlgorithm.getResult();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(logger::INFO,
                        "low_order_moments (native): master step took %d secs",
                        duration / 1000);

        /* Print the results */
        printNumericTable(result->get(low_order_moments::mean),
                          "low_order_moments first 20 columns of "
                          "Mean :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::variance),
                          "low_order_moments first 20 columns of "
                          "Variance :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::minimum),
                          "low_order_moments first 20 columns of "
                          "Minimum :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::maximum),
                          "low_order_moments first 20 columns of "
                          "Maximum :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::sum),
                          "low_order_moments first 20 columns of "
                          "Sum :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::sumSquares),
                          "low_order_moments first 20 columns of "
                          "SumSquares :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::sumSquaresCentered),
                          "low_order_moments first 20 columns of "
                          "SumSquaresCentered :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::secondOrderRawMoment),
                          "low_order_moments first 20 columns of "
                          "SecondOrderRawMoment :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::standardDeviation),
                          "low_order_moments first 20 columns of "
                          "StandardDeviation :",
                          1, 20);
        printNumericTable(result->get(low_order_moments::variation),
                          "low_order_moments first 20 columns of "
                          "Variation :",
                          1, 20);

        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID meanNumericTableField =
            env->GetFieldID(clazz, "meanNumericTable", "J");
        jfieldID varianceNumericTableField =
            env->GetFieldID(clazz, "varianceNumericTable", "J");
        jfieldID minimumNumericTableField =
            env->GetFieldID(clazz, "minimumNumericTable", "J");
        jfieldID maximumNumericTableField =
            env->GetFieldID(clazz, "maximumNumericTable", "J");

        NumericTablePtr *mean =
            new NumericTablePtr(result->get(low_order_moments::mean));
        NumericTablePtr *variance =
            new NumericTablePtr(result->get(low_order_moments::variance));
        NumericTablePtr *max =
            new NumericTablePtr(result->get(low_order_moments::maximum));
        NumericTablePtr *min =
            new NumericTablePtr(result->get(low_order_moments::minimum));

        env->SetLongField(resultObj, meanNumericTableField, (jlong)mean);
        env->SetLongField(resultObj, varianceNumericTableField,
                          (jlong)variance);
        env->SetLongField(resultObj, maximumNumericTableField, (jlong)max);
        env->SetLongField(resultObj, minimumNumericTableField, (jlong)min);
    }
}

#ifdef CPU_GPU_PROFILE
static void doSummarizerOneAPICompute(
    JNIEnv *env, jlong pNumTabData, jlong numRows, jlong numCols,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    logger::println(logger::INFO, "oneDAL (native): GPU compute start");
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable = *reinterpret_cast<homogen_table *>(
        createHomogenTableWithArrayPtr(pNumTabData, numRows, numCols,
                                       comm.get_queue())
            .get());

    const auto bs_desc = basic_statistics::descriptor<GpuAlgorithmFPType>{};
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto result_train = preview::compute(comm, bs_desc, htable);
    if (isRoot) {
        logger::println(logger::INFO, "Minimum");
        printHomegenTable(result_train.get_min());
        logger::println(logger::INFO, "Maximum");
        printHomegenTable(result_train.get_max());
        logger::println(logger::INFO, "Mean");
        printHomegenTable(result_train.get_mean());
        logger::println(logger::INFO, "Variation");
        printHomegenTable(result_train.get_variance());
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
                                                                         t1)
                .count();
        logger::println(logger::INFO,
                        "Summarizer (native): computing step took %d secs",
                        duration / 1000);
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
#endif

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_SummarizerDALImpl_cSummarizerTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jlong pNumTabData, jlong numRows,
    jlong numCols, jint executorNum, jint executorCores,
    jint computeDeviceOrdinal, jintArray gpuIdxArray, jstring ip_port,
    jobject resultObj) {
    logger::println(logger::INFO,
                    "oneDAL (native): use DPC++ kernels; device %s",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        ccl::communicator &cclComm = getComm();
        int rankId = cclComm.rank();
        NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        logger::println(logger::INFO,
                        "oneDAL (native): Number of CPU threads used %d",
                        nThreadsNew);
        doSummarizerDAALCompute(env, obj, rankId, cclComm, pData, executorNum,
                                resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case ComputeDevice::gpu: {
        logger::println(logger::INFO,
                        "oneDAL (native): use GPU kernels with rankid %d",
                        rank);
        const char *str = env->GetStringUTFChars(ip_port, nullptr);
        ccl::string ccl_ip_port(str);
        auto comm = createDalCommunicator(executorNum, rank, ccl_ip_port);

        doSummarizerOneAPICompute(env, pNumTabData, numRows, numCols, comm,
                                  resultObj);
        env->ReleaseStringUTFChars(ip_port, str);
        break;
    }
#endif
    default: {
        deviceError("Summarizer",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());
    }
    }
    return 0;
}
