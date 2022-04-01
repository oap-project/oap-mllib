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

#include "OneCCL.h"
#include "com_intel_oap_mllib_stat_CorrelationDALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef double algorithmFPType; /* Algorithm floating-point type */

static void correlation_compute(JNIEnv *env, jobject obj, int rankId,
                                ccl::communicator &comm,
                                const NumericTablePtr &pData, size_t nBlocks,
                                jobject resultObj) {
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
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "Correleation (native): local step took " << duration
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
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "Correleation (native): ccl_allgatherv took " << duration
              << " secs" << std::endl;
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
            std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        std::cout << "Correlation (native): master step took " << duration
                  << " secs" << std::endl;

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

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executor_num,
    jint executor_cores, jboolean use_gpu, jintArray gpu_idx_array,
    jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();

    const size_t nBlocks = executor_num;

    NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);

#ifdef CPU_GPU_PROFILE

    if (use_gpu) {
        int n_gpu = env->GetArrayLength(gpu_idx_array);
        cout << "oneDAL (native): use GPU kernels with " << n_gpu << " GPU(s)"
             << endl;

        jint *gpu_indices = env->GetIntArrayElements(gpu_idx_array, 0);

        int size = comm.size();
        auto assigned_gpu =
            getAssignedGPU(comm, size, rankId, gpu_indices, n_gpu);

        // Set SYCL context
        cl::sycl::queue queue(assigned_gpu);
        daal::services::SyclExecutionContext ctx(queue);
        daal::services::Environment::getInstance()->setDefaultExecutionContext(
            ctx);

        correlation_compute(env, obj, rankId, comm, pData, nBlocks, resultObj);

        env->ReleaseIntArrayElements(gpu_idx_array, gpu_indices, 0);
    } else
#endif
    {
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(
            executor_cores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        cout << "oneDAL (native): Number of CPU threads used: " << nThreadsNew
             << endl;

        correlation_compute(env, obj, rankId, comm, pData, nBlocks, resultObj);
    }

    return 0;
}