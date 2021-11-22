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
#include "org_apache_spark_mllib_stat_SummarizerDALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef double algorithmFPType; /* Algorithm floating-point type */

static void summarizer_compute(JNIEnv *env,
                                           jobject obj,
                                           int rankId,
                                           ccl::communicator &comm,
                                           const NumericTablePtr &pData,
                                           size_t nBlocks,
                                           jobject resultObj) {
   using daal::byte;
   auto t1 = std::chrono::high_resolution_clock::now();

   const bool isRoot = (rankId == ccl_root);

   low_order_moments::Distributed<step1Local, algorithmFPType> localAlgorithm;

  /* Set the input data set to the algorithm */
   localAlgorithm.input.set(low_order_moments::data, pData);

   /* Compute low_order_moments */
   localAlgorithm.compute();

   auto t2 = std::chrono::high_resolution_clock::now();
   auto duration =
       std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
   std::cout << "low_order_moments (native): local step took " << duration << " secs"
             << std::endl;

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
   std::cout << "low_order_moments (native): ccl_gather took " << duration << " secs"
             << std::endl;
   if (isRoot) {
       auto t1 = std::chrono::high_resolution_clock::now();
       /* Create an algorithm to compute covariance on the master node */
       low_order_moments::Distributed<step2Master, algorithmFPType> masterAlgorithm;

       for (size_t i = 0; i < nBlocks; i++) {
           /* Deserialize partial results from step 1 */
           OutputDataArchive dataArch(serializedData.get() +
                                          perNodeArchLength * i,
                                      perNodeArchLength);

           low_order_moments::PartialResultPtr dataForStep2FromStep1(new low_order_moments::PartialResult());
           dataForStep2FromStep1->deserialize(dataArch);

           /* Set local partial results as input for the master-node algorithm
           */
           masterAlgorithm.input.add(low_order_moments::partialResults,
                                  dataForStep2FromStep1);
       }

       /* Set the parameter to choose the type of the output matrix */
       masterAlgorithm.parameter.estimatesToCompute = low_order_moments::estimatesAll;

       /* Merge and finalizeCompute covariance decomposition on the master node */
       masterAlgorithm.compute();
       masterAlgorithm.finalizeCompute();

       /* Retrieve the algorithm results */
       low_order_moments::ResultPtr result = masterAlgorithm.getResult();
       auto t2 = std::chrono::high_resolution_clock::now();
       auto duration =
           std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
       std::cout << "low_order_moments (native): master step took " << duration << " secs"
               << std::endl;

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
       env->SetLongField(resultObj, varianceNumericTableField, (jlong)variance);
       env->SetLongField(resultObj, maximumNumericTableField, (jlong)max);
       env->SetLongField(resultObj, minimumNumericTableField, (jlong)min);

   }
}

/*
 * Class:     org_apache_spark_mllib_stat_CorrelationDALImpl
 * Method:    cCorrelationTrainDAL
 * Signature: (JJDDIILorg/apache/spark/ml/stat/CorrelationResult;)J
 */
JNIEXPORT jlong JNICALL
Java_org_apache_spark_mllib_stat_SummarizerDALImpl_cSummarizerTrainDAL(
    JNIEnv *env, jobject obj, jlong pNumTabData,
    jint executor_num, jint executor_cores, jboolean use_gpu, jintArray gpu_idx_array, jobject resultObj) {

    ccl::communicator &comm = getComm();
    size_t rankId = comm.rank();
    std::cout << " rankId : " << rankId << " ! "
                  << std::endl;

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

            summarizer_compute(
                  env, obj, rankId, comm, pData, nBlocks, resultObj);

            env->ReleaseIntArrayElements(gpu_idx_array, gpu_indices, 0);
        } else
    #endif
        {
            // Set number of threads for oneDAL to use for each rank
            services::Environment::getInstance()->setNumberOfThreads(executor_cores);

            int nThreadsNew =
                services::Environment::getInstance()->getNumberOfThreads();
            cout << "oneDAL (native): Number of CPU threads used: " << nThreadsNew
                 << endl;

            summarizer_compute(
                env, obj, rankId, comm, pData, nBlocks, resultObj);
        }

        return 0;
}
