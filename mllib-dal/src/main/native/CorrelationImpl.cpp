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
#include "oneapi/dal/algo/covariance.hpp"
#endif

#include "OneCCL.h"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"
#include "service.h"

#include "Logger.h"

using namespace std;
#ifdef CPU_GPU_PROFILE
namespace covariance_gpu = oneapi::dal::covariance;
#endif
using namespace daal;
using namespace daal::services;
namespace covariance_cpu = daal::algorithms::covariance;

static void doCorrelationDaalCompute(JNIEnv *env, jobject obj, size_t rankId,
                                     ccl::communicator &comm,
                                     const NumericTablePtr &pData,
                                     size_t nBlocks, jobject resultObj) {
    using daal::byte;
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool isRoot = (rankId == ccl_root);

    covariance_cpu::Distributed<step1Local, CpuAlgorithmFPType> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance_cpu::data, pData);

    /* Compute covariance */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    logger::println(logger::INFO,
                    "Correleation (native): local step took %d secs",
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
                    "Correleation (native): ccl_allgatherv took %d secs",
                    duration / 1000);
    if (isRoot) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /* Create an algorithm to compute covariance on the master node */
        covariance_cpu::Distributed<step2Master, CpuAlgorithmFPType>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            covariance_cpu::PartialResultPtr dataForStep2FromStep1(
                new covariance_cpu::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(covariance_cpu::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Set the parameter to choose the type of the output matrix */
        masterAlgorithm.parameter.outputMatrixType =
            covariance_cpu::correlationMatrix;

        /* Merge and finalizeCompute covariance decomposition on the master node
         */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance_cpu::ResultPtr result = masterAlgorithm.getResult();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(logger::INFO,
                        "Correleation (native): master step took %d secs",
                        duration / 1000);

        /* Print the results */
        printNumericTable(result->get(covariance_cpu::correlation),
                          "Correlation first 20 columns of "
                          "correlation matrix:",
                          1, 20);
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        NumericTablePtr *correlation =
            new NumericTablePtr(result->get(covariance_cpu::correlation));

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation);
    }
}

#ifdef CPU_GPU_PROFILE
static void doCorrelationOneAPICompute(
    JNIEnv *env, jlong pNumTabData, jlong numRows, jlong numCols,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    logger::println(logger::INFO, "oneDAL (native): GPU compute start");
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable = *reinterpret_cast<homogen_table *>(
        createHomogenTableWithArrayPtr(pNumTabData, numRows, numCols,
                                       comm.get_queue())
            .get());

    const auto cor_desc =
        covariance_gpu::descriptor<GpuAlgorithmFPType>{}.set_result_options(
            covariance_gpu::result_options::cor_matrix |
            covariance_gpu::result_options::means);
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto result_train = preview::compute(comm, cor_desc, htable);
    if (isRoot) {
        logger::println(logger::INFO, "Mean:");
        printHomegenTable(result_train.get_means());
        logger::println(logger::INFO, "Correlation:");
        printHomegenTable(result_train.get_cor_matrix());
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        logger::println(
            logger::INFO,
            "Correlation batch(native): computing step took %d secs.",
            duration / 1000);
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        HomogenTablePtr correlation =
            std::make_shared<homogen_table>(result_train.get_cor_matrix());
        saveHomogenTablePtrToVector(correlation);

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation.get());
    }
}
#endif

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jint rank, jlong pNumTabData, jlong numRows,
    jlong numCols, jint executorNum, jint executorCores,
    jint computeDeviceOrdinal, jintArray gpuIdxArray, jstring store_path,
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
        doCorrelationDaalCompute(env, obj, rankId, cclComm, pData, executorNum,
                                 resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case ComputeDevice::gpu: {
        logger::println(logger::INFO,
                        "oneDAL (native): use GPU kernels with rankid %d",
                        rank);

        const char* path = env->GetStringUTFChars(store_path, nullptr);
        ccl::string kvs_store_path(str);
        auto comm = createDalCommunicator(executorNum, rank, kvs_store_path);

        doCorrelationOneAPICompute(env, pNumTabData, numRows, numCols, comm,
                                   resultObj);
        env->ReleaseStringUTFChars(store_path, path);
        break;
    }
#endif
    default: {
        deviceError("Correlation",
                    ComputeDeviceString[computeDeviceOrdinal].c_str());
    }
    }
    return 0;
}
