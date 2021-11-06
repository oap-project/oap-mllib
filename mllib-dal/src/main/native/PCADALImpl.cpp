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
#include "com_intel_oap_mllib_feature_PCADALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef double algorithmFPType; /* Algorithm floating-point type */

static void doPCADALCompute(JNIEnv *env, jobject obj, int rankId,
                            ccl::communicator &comm, NumericTablePtr &pData,
                            int nBlocks, jobject resultObj) {
    auto t1 = std::chrono::high_resolution_clock::now();

    pca::Distributed<step1Local, algorithmFPType, pca::svdDense> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(pca::data, pData);

    /* Compute PCA decomposition */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "PCA (native): local step took " << duration << " secs"
              << std::endl;

    t1 = std::chrono::high_resolution_clock::now();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<daal::byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    serializedData = services::SharedPtr<daal::byte>(
        new daal::byte[perNodeArchLength * nBlocks]);

    daal::byte *nodeResults = new daal::byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);

    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "PCA (native): serializing partial results took " << duration
              << " secs" << std::endl;

    vector<size_t> recv_counts(nBlocks * perNodeArchLength);
    for (int i = 0; i < nBlocks; i++)
        recv_counts[i] = perNodeArchLength;

    cout << "PCA (native): ccl_allgatherv receiving "
         << perNodeArchLength * nBlocks << " bytes" << endl;

    t1 = std::chrono::high_resolution_clock::now();

    /* Transfer partial results to step 2 on the root node */
    // MPI_Gather(nodeResults, perNodeArchLength, MPI_CHAR,
    // serializedData.get(), perNodeArchLength, MPI_CHAR, ccl_root,
    // MPI_COMM_WORLD);
    ccl::allgatherv(nodeResults, perNodeArchLength, serializedData.get(),
                    recv_counts, ccl::datatype::uint8, comm)
        .wait();

    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "PCA (native): ccl_allgatherv took " << duration << " secs"
              << std::endl;

    if (rankId == ccl_root) {
        auto t1 = std::chrono::high_resolution_clock::now();

        /* Create an algorithm for principal component analysis using the
         * svdDense method on the master node */
        pca::Distributed<step2Master, algorithmFPType, pca::svdDense>
            masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            services::SharedPtr<pca::PartialResult<pca::svdDense>>
                dataForStep2FromStep1 =
                    services::SharedPtr<pca::PartialResult<pca::svdDense>>(
                        new pca::PartialResult<pca::svdDense>());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(pca::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute PCA decomposition on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        pca::ResultPtr result = masterAlgorithm.getResult();

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
        std::cout << "PCA (native): master step took " << duration << " secs"
                  << std::endl;

        /* Print the results */
        printNumericTable(result->get(pca::eigenvalues),
                          "First 10 eigenvalues with first 20 dimensions:", 10,
                          20);
        printNumericTable(result->get(pca::eigenvectors),
                          "First 10 eigenvectors with first 20 dimensions:", 10,
                          20);
        // Return all eigenvalues & eigenvectors

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID pcNumericTableField =
            env->GetFieldID(clazz, "pcNumericTable", "J");
        jfieldID explainedVarianceNumericTableField =
            env->GetFieldID(clazz, "explainedVarianceNumericTable", "J");

        NumericTablePtr *eigenvalues =
            new NumericTablePtr(result->get(pca::eigenvalues));
        NumericTablePtr *eigenvectors =
            new NumericTablePtr(result->get(pca::eigenvectors));

        env->SetLongField(resultObj, pcNumericTableField, (jlong)eigenvectors);
        env->SetLongField(resultObj, explainedVarianceNumericTableField,
                          (jlong)eigenvalues);
    }
}

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_feature_PCADALImpl_cPCATrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint k, jint executor_num,
    jint executor_cores, jboolean use_gpu, jintArray gpu_idx_array,
    jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();

    const size_t nBlocks = executor_num;

    NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
    // Source data already normalized
    pData->setNormalizationFlag(NumericTableIface::standardScoreNormalized);

#ifdef CPU_GPU_PROFILE
    if (use_gpu) {
        int n_gpu = env->GetArrayLength(gpu_idx_array);
        jint *gpu_indices = env->GetIntArrayElements(gpu_idx_array, 0);

        std::cout << "oneDAL (native): use GPU kernels with " << n_gpu
                  << " GPU(s)" << std::endl;

        int size = comm.size();
        auto assigned_gpu =
            getAssignedGPU(comm, size, rankId, gpu_indices, n_gpu);

        // Set SYCL context
        cl::sycl::queue queue(assigned_gpu);
        daal::services::SyclExecutionContext ctx(queue);
        daal::services::Environment::getInstance()->setDefaultExecutionContext(
            ctx);

        doPCADALCompute(env, obj, rankId, comm, pData, nBlocks, resultObj);

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

        doPCADALCompute(env, obj, rankId, comm, pData, nBlocks, resultObj);
    }

    return 0;
}
