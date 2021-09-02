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
#include <vector>

#include "OneCCL.h"
#include "org_apache_spark_ml_stat_CorrelationDALImpl.h"
#include "service.h"


using namespace std;
using namespace daal;
using namespace daal::algorithms;


typedef double algorithmFPType; /* Algorithm floating-point type */

/*
 * Class:     org_apache_spark_ml_stat_CorrelationDALImpl
 * Method:    cCorrelationTrainDAL
 * Signature: (JJDDIILorg/apache/spark/ml/stat/CorrelationResult;)J
 */

JNIEXPORT jlong JNICALL
Java_org_apache_spark_ml_stat_CorrelationDALImpl_cCorrelationTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData,
    jint executor_num, jint executor_cores, jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();
    std::cout << " rankId : " << rankId << " ! "
                  << std::endl;

    const size_t nBlocks = executor_num;

    NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);
    // Source data already normalized
    pData->setNormalizationFlag(NumericTableIface::standardScoreNormalized);

    // Set number of threads for oneDAL to use for each rank
    services::Environment::getInstance()->setNumberOfThreads(executor_cores);

    int nThreadsNew =
        services::Environment::getInstance()->getNumberOfThreads();
    cout << "oneDAL (native): Number of CPU threads used: " << nThreadsNew
         << endl;

    auto t1 = std::chrono::high_resolution_clock::now();
       
    const bool isRoot = (rankId == ccl_root);

    covariance::Distributed<step1Local> localAlgorithm;

   /* Set the input data set to the algorithm */
    localAlgorithm.input.set(covariance::data, pData);

    /* Compute covariance */
    localAlgorithm.compute();

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "Correleation (native): local step took " << duration << " secs"
              << std::endl;

    t1 = std::chrono::high_resolution_clock::now();

    /* Serialize partial results required by step 2 */
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    std::vector<uint64_t> aPerNodeArchLength(comm.size());
    std::vector<size_t> aReceiveCount(comm.size(), 1);
    /* Transfer archive length to the step 2 on the root node */
    ccl::allgatherv(&perNodeArchLength, 1, aPerNodeArchLength.data(), aReceiveCount, comm).wait();

    ByteBuffer serializedData;
    /* Calculate total archive length */
    int totalArchLength = 0;

    for (size_t i = 0; i < nBlocks; ++i)
    {
        totalArchLength += aPerNodeArchLength[i];
    }
    aReceiveCount[ccl_root] = totalArchLength;

    serializedData.resize(totalArchLength);


    ByteBuffer nodeResults(perNodeArchLength);
    dataArch.copyArchiveToArray(&nodeResults[0], perNodeArchLength);

    /* Transfer partial results to step 2 on the root node */
    ccl::allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0], aPerNodeArchLength, comm).wait();
    t2 = std::chrono::high_resolution_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "Correleation (native): ccl_allgatherv took " << duration << " secs"
              << std::endl;
    if (isRoot) {
            auto t1 = std::chrono::high_resolution_clock::now();
            /* Create an algorithm to compute covariance on the master node */
            covariance::Distributed<step2Master> masterAlgorithm;
            std::cout << "nBlocks :" << nBlocks
                  << std::endl;
            for (size_t i = 0, shift = 0; i < nBlocks; shift += aPerNodeArchLength[i], ++i) {
                /* Deserialize partial results from step 1 */
                OutputDataArchive dataArch(&serializedData[shift], aPerNodeArchLength[i]);
                covariance::PartialResultPtr dataForStep2FromStep1(new covariance::PartialResult());
                dataForStep2FromStep1->deserialize(dataArch);
                /* Set local partial results as input for the master-node algorithm
                */
                masterAlgorithm.input.add(covariance::partialResults,
                                        dataForStep2FromStep1);
            }
            /* Merge and finalizeCompute covariance decomposition on the master node */
            masterAlgorithm.compute();
            masterAlgorithm.finalizeCompute();

            /* Retrieve the algorithm results */
            covariance::ResultPtr result = masterAlgorithm.getResult();
            auto t2 = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
            std::cout << "Correlation (native): master step took " << duration << " secs"
                    << std::endl;

            /* Print the results */
            printNumericTable(result->get(covariance::correlation),
                            "Correlation first 20 columns of "
                            "correlation matrix:",
                            1, 20);

            // Return all covariance & mean

            // Get the class of the input object
            jclass clazz = env->GetObjectClass(resultObj);
            // Get Field references
            jfieldID correlationNumericTableField =
                env->GetFieldID(clazz, "correlationNumericTable", "J");


            NumericTablePtr *correlation =
                new NumericTablePtr(result->get(covariance::correlation));


            env->SetLongField(resultObj, correlationNumericTableField, (jlong)correlation);
        }

        return 0;
        
}