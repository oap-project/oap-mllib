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

using namespace std;
#ifdef CPU_GPU_PROFILE
using namespace oneapi::dal;
#else
using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
#endif

#ifdef CPU_ONLY_PROFILE
typedef double algorithmFPType; /* Algorithm floating-point type */

static void doCorrelationDaalCompute(JNIEnv *env, jobject obj, int rankId,
                                     ccl::communicator &comm,
                                     const NumericTablePtr &pData,
                                     size_t nBlocks, jobject resultObj) {
    using daal::byte;
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool isRoot = (rankId == ccl_root);

    covariance::Distributed<step1Local, algorithmFPType> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance::data, pData);

    /* Compute covariance */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "Correleation (native): local step took " << duration / 1000
              << " secs" << std::endl;

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
    std::cout << "Correleation (native): ccl_allgatherv took "
              << duration / 1000 << " secs" << std::endl;
    if (isRoot) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /* Create an algorithm to compute covariance on the master node */
        covariance::Distributed<step2Master, algorithmFPType> masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            covariance::PartialResultPtr dataForStep2FromStep1(
                new covariance::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(covariance::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Set the parameter to choose the type of the output matrix */
        masterAlgorithm.parameter.outputMatrixType =
            covariance::correlationMatrix;

        /* Merge and finalizeCompute covariance decomposition on the master node
         */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance::ResultPtr result = masterAlgorithm.getResult();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "Correlation (native): master step took "
                  << duration / 1000 << " secs" << std::endl;

        /* Print the results */
        printNumericTable(result->get(covariance::correlation),
                          "Correlation first 20 columns of "
                          "correlation matrix:",
                          1, 20);
        // Return all covariance & mean
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID correlationNumericTableField =
            env->GetFieldID(clazz, "correlationNumericTable", "J");

        NumericTablePtr *correlation =
            new NumericTablePtr(result->get(covariance::correlation));

        env->SetLongField(resultObj, correlationNumericTableField,
                          (jlong)correlation);
    }
}
#endif

#ifdef CPU_GPU_PROFILE
static void doCorrelationOneAPICompute(
    JNIEnv *env, jlong pNumTabData,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): GPU compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto cor_desc = covariance::descriptor{}.set_result_options(
        covariance::result_options::cor_matrix |
        covariance::result_options::means);
    auto t1 = std::chrono::high_resolution_clock::now();
    const auto result_train = preview::compute(comm, cor_desc, htable);
    if (isRoot) {
        std::cout << "Mean:\n" << result_train.get_means() << std::endl;
        std::cout << "Correlation:\n"
                  << result_train.get_cor_matrix() << std::endl;
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "Correlation batch(native): computing step took "
                  << duration / 1000 << " secs." << std::endl;
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
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executorNum,
    jint executorCores, jint computeDeviceOrdinal, jintArray gpuIdxArray,
    jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels "
              << "; device " << ComputeDeviceString[computeDeviceOrdinal]
              << std::endl;
    ccl::communicator &cclComm = getComm();
    int rankId = cclComm.rank();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
#ifdef CPU_ONLY_PROFILE
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        std::cout << "oneDAL (native): Number of CPU threads used"
                  << nThreadsNew << std::endl;
        doCorrelationDaalCompute(env, obj, rankId, cclComm, pData, executorNum,
                                 resultObj);
    }
#else
    case ComputeDevice::gpu: {
        int nGpu = env->GetArrayLength(gpuIdxArray);
        std::cout << "oneDAL (native): use GPU kernels with " << nGpu
                  << " GPU(s)"
                  << " rankid " << rankId << std::endl;

        jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);

        int size = cclComm.size();
        ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);

        auto queue =
            getAssignedGPU(device, cclComm, size, rankId, gpuIndices, nGpu);

        ccl::shared_ptr_class<ccl::kvs> &kvs = getKvs();
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                queue, size, rankId, kvs);
        doCorrelationOneAPICompute(env, pNumTabData, comm, resultObj);
    }
#endif
    }

    return 0;
}
