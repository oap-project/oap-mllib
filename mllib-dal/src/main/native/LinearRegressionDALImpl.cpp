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
#include <daal.h>
#include <iostream>
#include <oneapi/ccl.hpp>
#include <vector>

#include "OneCCL.h"
#include "org_apache_spark_ml_regression_LinearRegressionDALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef double algorithmFPType; /* Algorithm floating-point type */

const string trainDatasetFileNames[] = { "/root/workspace/jiangbo/onedal_oneccl/data/distributed/linear_regression_train_1.csv" };

static NumericTablePtr linearregression_compute(int rankId, ccl::communicator &comm,
                                      const NumericTablePtr &pData,
                                      const NumericTablePtr &pLabel,
                                      size_t nBlocks) {

#if 0
/*************************************************/
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource(trainDatasetFileNames[rankId], DataSource::notAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for training data and labels */
    NumericTablePtr trainData(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr trainDependentVariables(new HomogenNumericTable<>(nDependentVariables, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(trainData, trainDependentVariables));

    /* Retrieve the data from the input file */
    trainDataSource.loadDataBlock(mergedData.get());

    linear_regression::training::Distributed<step1Local> localAlgorithm;

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(linear_regression::training::data, trainData);
    localAlgorithm.input.set(linear_regression::training::dependentVariables, trainDependentVariables);
/*************************************************/
#else
    linear_regression::training::Distributed<step1Local> localAlgorithm;

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(linear_regression::training::data, pData);
    localAlgorithm.input.set(linear_regression::training::dependentVariables, pLabel);
#endif
    /* Train the multiple linear regression model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    std::cout << "perNodeArchLength: " << perNodeArchLength << std::endl;

    serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);

    byte * nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(), perNodeArchLength);  // 4 x "14016"

    std::cout << "gather" << std::endl;
    // std::cout << "comm.size(): " << comm.size() << std::endl;
    /* Transfer partial results to step 2 on the root node */
    // MPI_Gather(nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, ccl_root, MPI_COMM_WORLD);
    // ccl::allgatherv((int8_t *)nodeResults, perNodeArchLength, (int8_t *)(serializedData.get()), aReceiveCount, comm).wait();
    ccl::gather((int8_t *)nodeResults, perNodeArchLength, (int8_t *)(serializedData.get()), perNodeArchLength, comm).wait();

    delete[] nodeResults;

    std::cout << "if (rankId == ccl_root)" << std::endl;
    NumericTablePtr resultTable;
    if (rankId == ccl_root)
    {
        // std::cout << "build the final multiple linear regression model on the master node" << std::endl;
        /* Create an algorithm object to build the final multiple linear regression model on the master node */
        linear_regression::training::Distributed<step2Master> masterAlgorithm;

        std::cout << "nBlocks: " << nBlocks << std::endl;
        for (size_t i = 0; i < nBlocks; i++)
        {
            /* Deserialize partial results from step 1 */
            std::cout << "OutputDataArchive dataArch()" << std::endl;
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

            linear_regression::training::PartialResultPtr dataForStep2FromStep1 = linear_regression::training::PartialResultPtr(new linear_regression::training::PartialResult());
            std::cout << "dataForStep2FromStep1->deserialize()" << std::endl;
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local multiple linear regression model as input for the master-node algorithm */
            std::cout << "masterAlgorithm.input.add()" << std::endl;
            masterAlgorithm.input.add(linear_regression::training::partialModels, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the multiple linear regression model on the master node */
        std::cout << "masterAlgorithm.compute()" << std::endl;
        masterAlgorithm.compute();
        std::cout << "masterAlgorithm.finalizeCompute()" << std::endl;
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        std::cout << "Retrieve the algorithm results" << std::endl;
        linear_regression::training::ResultPtr trainingResult = masterAlgorithm.getResult();
        resultTable = trainingResult->get(linear_regression::training::model)->getBeta();
        printNumericTable(resultTable, "Linear Regression coefficients:");
    }
    return resultTable;
}

// TODO: set lambda for ridge regression
static NumericTablePtr ridgeregression_compute(int rankId, ccl::communicator &comm,
                                      const NumericTablePtr &pData,
                                      const NumericTablePtr &pLabel,
                                      size_t nBlocks) {
    ridge_regression::training::Distributed<step1Local> localAlgorithm;

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(ridge_regression::training::data, pData);
    localAlgorithm.input.set(ridge_regression::training::dependentVariables, pLabel);

    /* Train the multiple ridge regression model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();
    // std::cout << "perNodeArchLength: " << perNodeArchLength << std::endl;

    serializedData = services::SharedPtr<byte>(new byte[perNodeArchLength * nBlocks]);

    byte * nodeResults = new byte[perNodeArchLength];
    dataArch.copyArchiveToArray(nodeResults, perNodeArchLength);
    std::vector<size_t> aReceiveCount(comm.size(), perNodeArchLength);  // 4 x "14016"

    // std::cout << "gather" << std::endl;
    // std::cout << "comm.size(): " << comm.size() << std::endl;
    /* Transfer partial results to step 2 on the root node */
    // MPI_Gather(nodeResults, perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, ccl_root, MPI_COMM_WORLD);
    // ccl::allgatherv((int8_t *)nodeResults, perNodeArchLength, (int8_t *)(serializedData.get()), aReceiveCount, comm).wait();
    ccl::gather((int8_t *)nodeResults, perNodeArchLength, (int8_t *)(serializedData.get()), perNodeArchLength, comm).wait();

    delete[] nodeResults;

    // std::cout << "if (rankId == ccl_root)" << std::endl;
    NumericTablePtr resultTable;
    if (rankId == ccl_root)
    {
        // std::cout << "build the final multiple ridge regression model on the master node" << std::endl;
        /* Create an algorithm object to build the final multiple ridge regression model on the master node */
        ridge_regression::training::Distributed<step2Master> masterAlgorithm;

        for (size_t i = 0; i < nBlocks; i++)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

            ridge_regression::training::PartialResultPtr dataForStep2FromStep1 = ridge_regression::training::PartialResultPtr(new ridge_regression::training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local multiple ridge regression model as input for the master-node algorithm */
            masterAlgorithm.input.add(ridge_regression::training::partialModels, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the multiple ridge regression model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        ridge_regression::training::ResultPtr trainingResult = masterAlgorithm.getResult();
        resultTable = trainingResult->get(ridge_regression::training::model)->getBeta();
        printNumericTable(resultTable, "Ridge Regression coefficients:");
    }
    return resultTable;
}

/*
 * Class:     org_apache_spark_ml_regression_LinearRegressionDALImpl
 * Method:    cLRTrainDAL
 * Signature: (JJZDDIILorg/apache/spark/ml/regression/LiRResult;)J
 */
JNIEXPORT jlong JNICALL
Java_org_apache_spark_ml_regression_LinearRegressionDALImpl_cLRTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData, jlong pNumTabLabel,
    jboolean fitIntercept, jdouble regParam, jdouble elasticNetParam,
    jint executor_num, jint executor_cores, jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();

    NumericTablePtr pLabel = *((NumericTablePtr *)pNumTabLabel);
    NumericTablePtr pData = *((NumericTablePtr *)pNumTabData);

    // Set number of threads for oneDAL to use for each rank
    services::Environment::getInstance()->setNumberOfThreads(executor_cores);

    int nThreadsNew = services::Environment::getInstance()->getNumberOfThreads();
    cout << "oneDAL (native): Number of CPU threads used: " << nThreadsNew << endl;

    NumericTablePtr resultTable;
    // TODO: get condition from regParam and elasticNetParam
    if (true) {
        // linear regression
        resultTable = linearregression_compute(rankId, comm, pData, pLabel, executor_num);
    } else {
        // ridge regression
        resultTable = ridgeregression_compute(rankId, comm, pData, pLabel, executor_num);
    }

    cout << "Fill resultObj" << endl;
    if (rankId == ccl_root)
    {
        // Get the class of the result object
        jclass clazz = env->GetObjectClass(resultObj);

        // Get Field references
        jfieldID interceptField = env->GetFieldID(clazz, "intercept", "D");
        jfieldID coeffNumericTableField = env->GetFieldID(clazz, "coeffNumericTable", "J");

        NumericTablePtr *coeffvectors = new NumericTablePtr(resultTable);
        env->SetLongField(resultObj, coeffNumericTableField, (jlong)coeffvectors);

        //TODO: set intercept

        return (jlong)coeffvectors;
    } else
        return (jlong)0;

}
