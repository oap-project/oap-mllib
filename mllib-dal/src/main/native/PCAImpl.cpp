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
#include "Common.hpp"
#include "oneapi/dal/algo/covariance.hpp"
#include "oneapi/dal/algo/pca.hpp"
#endif

#include "OneCCL.h"
#include "com_intel_oap_mllib_feature_PCADALImpl.h"
#include "service.h"

using namespace std;
#ifdef CPU_GPU_PROFILE
namespace pca_gpu = oneapi::dal::pca;
namespace covariance_gpu = oneapi::dal::covariance;
#endif
using namespace daal;
using namespace daal::services;
namespace pca_cpu = daal::algorithms::pca;
namespace covariance_cpu = daal::algorithms::covariance;

typedef double algorithmFPType; /* Algorithm floating-point type */

static void doPCADAALCompute(JNIEnv *env, jobject obj, size_t rankId,
                             ccl::communicator &comm, NumericTablePtr &pData,
                             size_t nBlocks, jobject resultObj) {
    std::cout << "oneDAL (native): CPU compute start" << std::endl;
    using daal::byte;
    auto t1 = std::chrono::high_resolution_clock::now();

    const bool isRoot = (rankId == ccl_root);

    covariance_cpu::Distributed<step1Local, algorithmFPType> localAlgorithm;

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance_cpu::data, pData);

    /* Compute covariance for PCA*/
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << " PCA (native): Covariance local step took " << duration / 1000
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
    std::vector<size_t> aReceiveCount(comm.size(), perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    ccl::gather((int8_t *)nodeResults, perNodeArchLength,
                (int8_t *)(serializedData.get()), perNodeArchLength, comm)
        .wait();
    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "PCA (native): Covariance gather to master took "
              << duration / 1000 << " secs" << std::endl;
    if (isRoot) {
        auto t1 = std::chrono::high_resolution_clock::now();
        /* Create an algorithm to compute covariance on the master node */
        covariance_cpu::Distributed<step2Master, algorithmFPType>
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
            covariance_cpu::covarianceMatrix;

        /* Merge and finalizeCompute covariance decomposition on the master node
         */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        covariance_cpu::ResultPtr covariance_result =
            masterAlgorithm.getResult();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "PCA (native): Covariance master step took "
                  << duration / 1000 << " secs" << std::endl;

        t1 = std::chrono::high_resolution_clock::now();

        /* Create an algorithm for principal component analysis using the
         * correlation method*/
        pca_cpu::Batch<algorithmFPType> algorithm;

        /* Set the algorithm input data*/
        algorithm.input.set(pca_cpu::correlation,
                            covariance_result->get(covariance_cpu::covariance));
        algorithm.parameter.resultsToCompute = pca_cpu::eigenvalue;

        /* Compute results of the PCA algorithm*/
        algorithm.compute();

        t2 = std::chrono::high_resolution_clock::now();
        duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "PCA (native): master step took " << duration / 1000
                  << " secs" << std::endl;

        /* Print the results */
        pca_cpu::ResultPtr result = algorithm.getResult();
        printNumericTable(result->get(pca_cpu::eigenvalues),
                          "First 10 eigenvalues with first 20 dimensions:", 10,
                          20);
        printNumericTable(result->get(pca_cpu::eigenvectors),
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
            new NumericTablePtr(result->get(pca_cpu::eigenvalues));
        NumericTablePtr *eigenvectors =
            new NumericTablePtr(result->get(pca_cpu::eigenvectors));

        env->SetLongField(resultObj, pcNumericTableField, (jlong)eigenvectors);
        env->SetLongField(resultObj, explainedVarianceNumericTableField,
                          (jlong)eigenvalues);
    }
}

#ifdef CPU_GPU_PROFILE
static void doPCAOneAPICompute(
    JNIEnv *env, jlong pNumTabData,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): GPU compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);

    const auto cov_desc = covariance_gpu::descriptor{}.set_result_options(
        covariance_gpu::result_options::cov_matrix);

    auto t1 = std::chrono::high_resolution_clock::now();
    const auto result = preview::compute(comm, cov_desc, htable);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "PCA (native): Covariance step took " << duration / 1000
              << " secs" << std::endl;
    if (isRoot) {
        using float_t = double;
        using method_t = pca_gpu::method::precomputed;
        using task_t = pca_gpu::task::dim_reduction;
        using descriptor_t = pca_gpu::descriptor<float_t, method_t, task_t>;
        const auto pca_desc = descriptor_t().set_deterministic(true);

        t1 = std::chrono::high_resolution_clock::now();
        const auto result_train =
            preview::train(comm, pca_desc, result.get_cov_matrix());
        t2 = std::chrono::high_resolution_clock::now();
        duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "PCA (native): Eigen step took " << duration / 1000
                  << " secs." << std::endl;
        // Return all eigenvalues & eigenvectors
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID pcNumericTableField =
            env->GetFieldID(clazz, "pcNumericTable", "J");
        jfieldID explainedVarianceNumericTableField =
            env->GetFieldID(clazz, "explainedVarianceNumericTable", "J");
        std::cout << "Eigenvectors:\n"
                  << result_train.get_eigenvectors() << std::endl;
        std::cout << "Eigenvalues:\n"
                  << result_train.get_eigenvalues() << std::endl;

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
#endif

JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_feature_PCADALImpl_cPCATrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jint executorNum,
    jint executorCores, jint computeDeviceOrdinal, jintArray gpuIdxArray,
    jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels "
              << "; device " << ComputeDeviceString[computeDeviceOrdinal]
              << std::endl;
    ccl::communicator &cclComm = getComm();
    size_t rankId = cclComm.rank();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        std::cout << "oneDAL (native): Number of CPU threads used "
                  << nThreadsNew << std::endl;
        doPCADAALCompute(env, obj, rankId, cclComm, pData, executorNum,
                         resultObj);
        break;
    }
#ifdef CPU_GPU_PROFILE
    case ComputeDevice::gpu: {
        int nGpu = env->GetArrayLength(gpuIdxArray);
        std::cout << "oneDAL (native): use GPU kernels with " << nGpu
                  << " GPU(s)"
                  << " rankid " << rankId << std::endl;

        jint *gpuIndices = env->GetIntArrayElements(gpuIdxArray, 0);

        int size = cclComm.size();

        auto queue =
            getAssignedGPU(device, cclComm, size, rankId, gpuIndices, nGpu);

        ccl::shared_ptr_class<ccl::kvs> &kvs = getKvs();
        auto comm =
            preview::spmd::make_communicator<preview::spmd::backend::ccl>(
                queue, size, rankId, kvs);
        doPCAOneAPICompute(env, pNumTabData, comm, resultObj);
        env->ReleaseIntArrayElements(gpuIdxArray, gpuIndices, 0);
        break;
    }
#endif
    default: {
        std::cout << "no supported device!" << std::endl;
        exit(-1);
    }
    }
    return 0;
}
