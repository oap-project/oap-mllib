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
#include <iomanip>
#include <iostream>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "Communicator.hpp"
#include "OutputHelpers.hpp"
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/table/homogen.hpp"
#endif

#include "OneCCL.h"
#include "com_intel_oap_mllib_clustering_KMeansDALImpl.h"
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

static NumericTablePtr kmeans_compute(int rankId, ccl::communicator &comm,
                                      const NumericTablePtr &pData,
                                      const NumericTablePtr &initialCentroids,
                                      size_t nClusters, size_t nBlocks,
                                      algorithmFPType &ret_cost) {
    const bool isRoot = (rankId == ccl_root);
    size_t CentroidsArchLength = 0;
    InputDataArchive inputArch;
    if (isRoot) {
        /* Retrieve the algorithm results and serialize them */
        initialCentroids->serialize(inputArch);
        CentroidsArchLength = inputArch.getSizeOfArchive();
    }

    /* Get partial results from the root node */
    ccl::broadcast(&CentroidsArchLength, sizeof(size_t), ccl::datatype::uint8,
                   ccl_root, comm)
        .wait();

    ByteBuffer nodeCentroids(CentroidsArchLength);
    if (isRoot)
        inputArch.copyArchiveToArray(&nodeCentroids[0], CentroidsArchLength);

    ccl::broadcast(&nodeCentroids[0], CentroidsArchLength, ccl::datatype::uint8,
                   ccl_root, comm)
        .wait();

    /* Deserialize centroids data */
    OutputDataArchive outArch(nodeCentroids.size() ? &nodeCentroids[0] : NULL,
                              CentroidsArchLength);

    NumericTablePtr centroids(new HomogenNumericTable<algorithmFPType>());

    centroids->deserialize(outArch);

    /* Create an algorithm to compute k-means on local nodes */
    kmeans::Distributed<step1Local, algorithmFPType> localAlgorithm(nClusters);

    /* Set the input data set to the algorithm */
    localAlgorithm.input.set(kmeans::data, pData);
    localAlgorithm.input.set(kmeans::inputCentroids, centroids);

    /* Compute k-means */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    ByteBuffer serializedData;

    /* Serialized data is of equal size on each node if each node called
     * compute() equal number of times */
    vector<size_t> recvCounts(nBlocks);
    for (size_t i = 0; i < nBlocks; i++) {
        recvCounts[i] = perNodeArchLength;
    }
    serializedData.resize(perNodeArchLength * nBlocks);

    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    ccl::allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0],
                    recvCounts, ccl::datatype::uint8, comm)
        .wait();

    if (isRoot) {
        /* Create an algorithm to compute k-means on the master node */
        kmeans::Distributed<step2Master, algorithmFPType> masterAlgorithm(
            nClusters);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&serializedData[perNodeArchLength * i],
                                       perNodeArchLength);

            kmeans::PartialResultPtr dataForStep2FromStep1(
                new kmeans::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set local partial results as input for the master-node algorithm
             */
            masterAlgorithm.input.add(kmeans::partialResults,
                                      dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute k-means on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        ret_cost = masterAlgorithm.getResult()
                       ->get(kmeans::objectiveFunction)
                       ->getValue<algorithmFPType>(0, 0);

        /* Retrieve the algorithm results */
        return masterAlgorithm.getResult()->get(kmeans::centroids);
    }
    return NumericTablePtr();
}

static bool isCenterConverged(const algorithmFPType *oldCenter,
                              const algorithmFPType *newCenter, size_t dim,
                              double tolerance) {

    algorithmFPType sums = 0.0;

    for (size_t i = 0; i < dim; i++)
        sums += (newCenter[i] - oldCenter[i]) * (newCenter[i] - oldCenter[i]);

    return sums <= tolerance * tolerance;
}

static bool areAllCentersConverged(const NumericTablePtr &oldCenters,
                                   const NumericTablePtr &newCenters,
                                   double tolerance) {
    size_t rows = oldCenters->getNumberOfRows();
    size_t cols = oldCenters->getNumberOfColumns();

    BlockDescriptor<algorithmFPType> blockOldCenters;
    oldCenters->getBlockOfRows(0, rows, readOnly, blockOldCenters);
    algorithmFPType *arrayOldCenters = blockOldCenters.getBlockPtr();

    BlockDescriptor<algorithmFPType> blockNewCenters;
    newCenters->getBlockOfRows(0, rows, readOnly, blockNewCenters);
    algorithmFPType *arrayNewCenters = blockNewCenters.getBlockPtr();

    for (size_t i = 0; i < rows; i++) {
        if (!isCenterConverged(&arrayOldCenters[i * cols],
                               &arrayNewCenters[i * cols], cols, tolerance))
            return false;
    }

    return true;
}

static jlong doKMeansDaalCompute(JNIEnv *env, jobject obj, int rankId,
                                 ccl::communicator &comm,
                                 NumericTablePtr &pData,
                                 NumericTablePtr &centroids, jint cluster_num,
                                 jdouble tolerance, jint iteration_num,
                                 jint executor_num, jobject resultObj) {
    std::cout << "oneDAL (native): CPU compute start" << std::endl;
    algorithmFPType totalCost;

    NumericTablePtr newCentroids;
    bool converged = false;

    int it = 0;
    for (it = 0; it < iteration_num && !converged; it++) {
        auto t1 = std::chrono::high_resolution_clock::now();

        newCentroids = kmeans_compute(rankId, comm, pData, centroids,
                                      cluster_num, executor_num, totalCost);

        if (rankId == ccl_root) {
            converged =
                areAllCentersConverged(centroids, newCentroids, tolerance);
        }

        // Sync converged status
        ccl::broadcast(&converged, 1, ccl::datatype::uint8, ccl_root, comm)
            .wait();

        centroids = newCentroids;

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "KMeans (native): iteration " << it << " took "
                  << duration / 1000 << " secs" << std::endl;
    }

    if (rankId == ccl_root) {
        if (it == iteration_num)
            std::cout << "KMeans (native): reached " << iteration_num
                      << " max iterations." << std::endl;
        else
            std::cout << "KMeans (native): converged in " << it
                      << " iterations." << std::endl;

        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");

        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField, it);
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField, totalCost);

        NumericTablePtr *ret = new NumericTablePtr(centroids);
        return (jlong)ret;
    } else {
        return (jlong)0;
    }
}
#endif

#ifdef CPU_GPU_PROFILE
static jlong doKMeansOneAPICompute(
    JNIEnv *env, jlong pNumTabData, jlong pNumTabCenters, jint clusterNum,
    jdouble tolerance, jint iterationNum,
    preview::spmd::communicator<preview::spmd::device_memory_access::usm> comm,
    jobject resultObj) {
    std::cout << "oneDAL (native): GPU compute start" << std::endl;
    const bool isRoot = (comm.get_rank() == ccl_root);
    homogen_table htable =
        *reinterpret_cast<const homogen_table *>(pNumTabData);
    homogen_table centroids =
        *reinterpret_cast<const homogen_table *>(pNumTabCenters);
    const auto kmeans_desc = kmeans::descriptor<>()
                                 .set_cluster_count(clusterNum)
                                 .set_max_iteration_count(iterationNum)
                                 .set_accuracy_threshold(tolerance);
    kmeans::train_input local_input{htable, centroids};
    auto t1 = std::chrono::high_resolution_clock::now();
    kmeans::train_result result_train =
        preview::train(comm, kmeans_desc, local_input);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "KMeans (native): training step took " << duration / 1000
              << " secs" << std::endl;
    if (isRoot) {
        std::cout << "Iteration count: " << result_train.get_iteration_count()
                  << std::endl;
        std::cout << "Centroids:\n"
                  << result_train.get_model().get_centroids() << std::endl;
        t2 = std::chrono::high_resolution_clock::now();
        duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count();
        std::cout << "KMeans (native): training step took " << duration / 1000
                  << " secs in end." << std::endl;
        // Get the class of the input object
        jclass clazz = env->GetObjectClass(resultObj);
        // Get Field references
        jfieldID totalCostField = env->GetFieldID(clazz, "totalCost", "D");
        jfieldID iterationNumField =
            env->GetFieldID(clazz, "iterationNum", "I");
        // Set iteration num for result
        env->SetIntField(resultObj, iterationNumField,
                         result_train.get_iteration_count());
        // Set cost for result
        env->SetDoubleField(resultObj, totalCostField,
                            result_train.get_objective_function_value());

        HomogenTablePtr centroidsPtr = std::make_shared<homogen_table>(
            result_train.get_model().get_centroids());
        saveHomogenTablePtrToVector(centroidsPtr);
        return (jlong)centroidsPtr.get();
    } else {
        return (jlong)0;
    }
}
#endif

/*
 * Class:     com_intel_oap_mllib_clustering_KMeansDALImpl
 * Method:    cKMeansOneapiComputeWithInitCenters
 * Signature: (JJIDIIILcom/intel/oap/mllib/clustering/KMeansResult;)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oap_mllib_clustering_KMeansDALImpl_cKMeansOneapiComputeWithInitCenters(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabCenters,
    jint clusterNum, jdouble tolerance, jint iterationNum, jint executorNum,
    jint executorCores, jint computeDeviceOrdinal, jintArray gpuIdxArray,
    jobject resultObj) {
    std::cout << "oneDAL (native): use DPC++ kernels "
              << "; device " << ComputeDeviceString[computeDeviceOrdinal]
              << std::endl;
    jlong ret = 0L;
    ccl::communicator &cclComm = getComm();
    int rankId = cclComm.rank();
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch (device) {
#ifdef CPU_ONLY_PROFILE
    case ComputeDevice::host:
    case ComputeDevice::cpu: {
        NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
        NumericTablePtr centroids = *((NumericTablePtr *)pNumTabCenters);
        // Set number of threads for oneDAL to use for each rank
        services::Environment::getInstance()->setNumberOfThreads(executorCores);

        int nThreadsNew =
            services::Environment::getInstance()->getNumberOfThreads();
        std::cout << "oneDAL (native): Number of CPU threads used "
                  << nThreadsNew << std::endl;
        ret = doKMeansDaalCompute(env, obj, rankId, cclComm, pData, centroids,
                                  clusterNum, tolerance, iterationNum,
                                  executorNum, resultObj);
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
        ret =
            doKMeansOneAPICompute(env, pNumTabData, pNumTabCenters, clusterNum,
                                  tolerance, iterationNum, comm, resultObj);
    }
#endif
    }
    return ret;
}
