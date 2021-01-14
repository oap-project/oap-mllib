#include <assert.h>
#include <ccl.h>
#include <daal.h>

#include <chrono>
#include <iostream>

#include "ALSShuffle.h"
#include "org_apache_spark_ml_recommendation_ALSDALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::implicit_als;

const int ccl_root = 0;

typedef float algorithmFPType; /* Algorithm floating-point type */

NumericTablePtr userOffset;
NumericTablePtr itemOffset;

// KeyValueDataCollectionPtr userOffsetsOnMaster;
// KeyValueDataCollectionPtr itemOffsetsOnMaster;

CSRNumericTablePtr dataTable;
CSRNumericTablePtr transposedDataTable;

KeyValueDataCollectionPtr userStep3LocalInput;
KeyValueDataCollectionPtr itemStep3LocalInput;

training::DistributedPartialResultStep4Ptr itemsPartialResultLocal;
training::DistributedPartialResultStep4Ptr usersPartialResultLocal;
std::vector<training::DistributedPartialResultStep4Ptr> itemsPartialResultsMaster;
std::vector<training::DistributedPartialResultStep4Ptr> usersPartialResultsMaster;

template <typename T>
void gather(size_t rankId, size_t nBlocks, const ByteBuffer& nodeResults, T* result) {
  size_t perNodeArchLengthMaster[nBlocks];
  size_t perNodeArchLength = nodeResults.size();
  ByteBuffer serializedData;
  ccl_request_t request;

  size_t recv_counts[nBlocks];
  for (size_t i = 0; i < nBlocks; i++) recv_counts[i] = sizeof(size_t);

  // MPI_Gather(&perNodeArchLength, sizeof(int), MPI_CHAR, perNodeArchLengthMaster,
  // sizeof(int), MPI_CHAR, ccl_root, MPI_COMM_WORLD);
  ccl_allgatherv(&perNodeArchLength, sizeof(size_t), perNodeArchLengthMaster, recv_counts,
                 ccl_dtype_char, NULL, NULL, NULL, &request);
  ccl_wait(request);

  // should resize for all ranks for ccl_allgatherv
  size_t memoryBuf = 0;
  for (size_t i = 0; i < nBlocks; i++) {
    memoryBuf += perNodeArchLengthMaster[i];
  }
  serializedData.resize(memoryBuf);

  std::vector<int> displs(nBlocks);
  if (rankId == ccl_root) {
    size_t shift = 0;
    for (size_t i = 0; i < nBlocks; i++) {
      displs[i] = shift;
      shift += perNodeArchLengthMaster[i];
    }
  }

  /* Transfer partial results to step 2 on the root node */
  // MPI_Gatherv(&nodeResults[0], perNodeArchLength, MPI_CHAR, &serializedData[0],
  // perNodeArchLengthMaster, displs, MPI_CHAR, ccl_root,
  //             MPI_COMM_WORLD);
  ccl_allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0],
                 perNodeArchLengthMaster, ccl_dtype_char, NULL, NULL, NULL, &request);
  ccl_wait(request);

  if (rankId == ccl_root) {
    for (size_t i = 0; i < nBlocks; i++) {
      /* Deserialize partial results from step 1 */
      result[i] = result[i]->cast(deserializeDAALObject(&serializedData[0] + displs[i],
                                                        perNodeArchLengthMaster[i]));
    }
  }
}

// void gatherUsers(const ByteBuffer & nodeResults, int nBlocks)
// {
//     size_t perNodeArchLengthMaster[nBlocks];
//     size_t perNodeArchLength = nodeResults.size();
//     ByteBuffer serializedData;
//     size_t recv_counts[nBlocks];
//     for (int i = 0; i < nBlocks; i++) {
//         recv_counts[i] = sizeof(size_t);
//     }

//     ccl_request_t request;
//     // MPI_Allgather(&perNodeArchLength, sizeof(int), MPI_CHAR,
//     perNodeArchLengthMaster, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);
//     ccl_allgatherv(&perNodeArchLength, sizeof(size_t), perNodeArchLengthMaster,
//     recv_counts, ccl_dtype_char, NULL, NULL, NULL, &request); ccl_wait(request);

//     size_t memoryBuf = 0;
//     for (int i = 0; i < nBlocks; i++)
//     {
//         memoryBuf += perNodeArchLengthMaster[i];
//     }
//     serializedData.resize(memoryBuf);

//     size_t shift = 0;
//     std::vector<int> displs(nBlocks);
//     for (int i = 0; i < nBlocks; i++)
//     {
//         displs[i] = shift;
//         shift += perNodeArchLengthMaster[i];
//     }

//     /* Transfer partial results to step 2 on the root node */
//     // MPI_Allgatherv(&nodeResults[0], perNodeArchLength, MPI_CHAR, &serializedData[0],
//     perNodeArchLengthMaster, displs, MPI_CHAR, MPI_COMM_WORLD);
//     ccl_allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0],
//     perNodeArchLengthMaster, ccl_dtype_char, NULL, NULL, NULL, &request);
//     ccl_wait(request);

//     usersPartialResultsMaster.resize(nBlocks);
//     for (int i = 0; i < nBlocks; i++)
//     {
//         /* Deserialize partial results from step 4 */
//         usersPartialResultsMaster[i] =
//             training::DistributedPartialResultStep4::cast(deserializeDAALObject(&serializedData[0]
//             + displs[i], perNodeArchLengthMaster[i]));
//     }
// }

// void gatherItems(const ByteBuffer & nodeResults, size_t nBlocks)
// {
//     size_t perNodeArchLengthMaster[nBlocks];
//     size_t perNodeArchLength = nodeResults.size();
//     ByteBuffer serializedData;
//     size_t recv_counts[nBlocks];
//     for (size_t i = 0; i < nBlocks; i++) {
//         recv_counts[i] = sizeof(size_t);
//     }

//     ccl_request_t request;
//     // MPI_Allgather(&perNodeArchLength, sizeof(int), MPI_CHAR,
//     perNodeArchLengthMaster, sizeof(int), MPI_CHAR, MPI_COMM_WORLD);
//     ccl_allgatherv(&perNodeArchLength, sizeof(size_t), perNodeArchLengthMaster,
//     recv_counts, ccl_dtype_char, NULL, NULL, NULL, &request); ccl_wait(request);

//     size_t memoryBuf = 0;
//     for (size_t i = 0; i < nBlocks; i++)
//     {
//         memoryBuf += perNodeArchLengthMaster[i];
//     }
//     serializedData.resize(memoryBuf);

//     size_t shift = 0;
//     std::vector<int> displs(nBlocks);
//     for (size_t i = 0; i < nBlocks; i++)
//     {
//         displs[i] = shift;
//         shift += perNodeArchLengthMaster[i];
//     }

//     /* Transfer partial results to step 2 on the root node */
//     // MPI_Allgatherv(&nodeResults[0], perNodeArchLength, MPI_CHAR, &serializedData[0],
//     perNodeArchLengthMaster, displs, MPI_CHAR, MPI_COMM_WORLD);
//     ccl_allgatherv(&nodeResults[0], perNodeArchLength, &serializedData[0],
//     perNodeArchLengthMaster, ccl_dtype_char, NULL, NULL, NULL, &request);
//     ccl_wait(request);

//     itemsPartialResultsMaster.resize(nBlocks);
//     for (size_t i = 0; i < nBlocks; i++)
//     {
//         /* Deserialize partial results from step 4 */
//         itemsPartialResultsMaster[i] =
//             training::DistributedPartialResultStep4::cast(deserializeDAALObject(&serializedData[0]
//             + displs[i], perNodeArchLengthMaster[i]));
//     }
// }

template <typename T>
void all2all(ByteBuffer* nodeResults, size_t nBlocks, KeyValueDataCollectionPtr result) {
  size_t memoryBuf = 0;
  size_t shift = 0;
  size_t perNodeArchLengths[nBlocks];
  size_t perNodeArchLengthsRecv[nBlocks];
  std::vector<size_t> sdispls(nBlocks);
  ByteBuffer serializedSendData;
  ByteBuffer serializedRecvData;

  for (size_t i = 0; i < nBlocks; i++) {
    perNodeArchLengths[i] = nodeResults[i].size();
    memoryBuf += perNodeArchLengths[i];
    sdispls[i] = shift;
    shift += perNodeArchLengths[i];
  }
  serializedSendData.resize(memoryBuf);

  /* memcpy to avoid double compute */
  memoryBuf = 0;
  for (size_t i = 0; i < nBlocks; i++) {
    for (size_t j = 0; j < perNodeArchLengths[i]; j++)
      serializedSendData[memoryBuf + j] = nodeResults[i][j];
    memoryBuf += perNodeArchLengths[i];
  }

  ccl_request_t request;
  // MPI_Alltoall(perNodeArchLengths, sizeof(int), MPI_CHAR, perNodeArchLengthsRecv,
  // sizeof(int), MPI_CHAR, MPI_COMM_WORLD);
  ccl_alltoall(perNodeArchLengths, perNodeArchLengthsRecv, sizeof(size_t), ccl_dtype_char,
               NULL, NULL, NULL, &request);
  ccl_wait(request);

  memoryBuf = 0;
  shift = 0;
  std::vector<size_t> rdispls(nBlocks);
  for (size_t i = 0; i < nBlocks; i++) {
    memoryBuf += perNodeArchLengthsRecv[i];
    rdispls[i] = shift;
    shift += perNodeArchLengthsRecv[i];
  }

  serializedRecvData.resize(memoryBuf);

  /* Transfer partial results to step 2 on the root node */
  // MPI_Alltoallv(&serializedSendData[0], perNodeArchLengths, sdispls, MPI_CHAR,
  // &serializedRecvData[0], perNodeArchLengthsRecv, rdispls, MPI_CHAR,
  //               MPI_COMM_WORLD);
  ccl_alltoallv(&serializedSendData[0], perNodeArchLengths, &serializedRecvData[0],
                perNodeArchLengthsRecv, ccl_dtype_char, NULL, NULL, NULL, &request);
  ccl_wait(request);

  for (size_t i = 0; i < nBlocks; i++) {
    (*result)[i] = T::cast(deserializeDAALObject(&serializedRecvData[rdispls[i]],
                                                 perNodeArchLengthsRecv[i]));
  }
}

KeyValueDataCollectionPtr initializeStep1Local(size_t rankId, size_t partitionId,
                                               size_t nBlocks, size_t nUsers,
                                               size_t nFactors) {
  int usersPartition[1] = {(int)nBlocks};

  /* Create an algorithm object to initialize the implicit ALS model with the default
   * method */
  training::init::Distributed<step1Local, algorithmFPType, training::init::fastCSR>
      initAlgorithm;
  initAlgorithm.parameter.fullNUsers = nUsers;
  initAlgorithm.parameter.nFactors = nFactors;
  initAlgorithm.parameter.seed += rankId;
  initAlgorithm.parameter.partition.reset(
      new HomogenNumericTable<int>((int*)usersPartition, 1, 1));
  /* Pass a training data set and dependent values to the algorithm */
  initAlgorithm.input.set(training::init::data, dataTable);

  /* Initialize the implicit ALS model */
  initAlgorithm.compute();

  training::init::PartialResultPtr partialResult = initAlgorithm.getPartialResult();
  itemStep3LocalInput = partialResult->get(training::init::outputOfInitForComputeStep3);
  userOffset = partialResult->get(training::init::offsets, (size_t)rankId);
  // if (rankId == ccl_root)
  // {
  //     userOffsetsOnMaster = partialResult->get(training::init::offsets);
  // }
  PartialModelPtr partialModelLocal = partialResult->get(training::init::partialModel);

  itemsPartialResultLocal.reset(new training::DistributedPartialResultStep4());
  itemsPartialResultLocal->set(training::outputOfStep4ForStep1, partialModelLocal);

  return partialResult->get(training::init::outputOfStep1ForStep2);
}

void initializeStep2Local(size_t rankId, size_t partitionId,
                          const KeyValueDataCollectionPtr& initStep2LocalInput) {
  /* Create an algorithm object to perform the second step of the implicit ALS
   * initialization algorithm */
  training::init::Distributed<step2Local, algorithmFPType, training::init::fastCSR>
      initAlgorithm;

  initAlgorithm.input.set(training::init::inputOfStep2FromStep1, initStep2LocalInput);

  /* Compute partial results of the second step on local nodes */
  initAlgorithm.compute();

  training::init::DistributedPartialResultStep2Ptr partialResult =
      initAlgorithm.getPartialResult();
  transposedDataTable =
      CSRNumericTable::cast(partialResult->get(training::init::transposedData));
  userStep3LocalInput = partialResult->get(training::init::outputOfInitForComputeStep3);
  itemOffset = partialResult->get(training::init::offsets, (size_t)rankId);
  // if (rankId == ccl_root)
  // {
  //     itemOffsetsOnMaster = partialResult->get(training::init::offsets);
  // }
}

void initializeModel(size_t rankId, size_t partitionId, size_t nBlocks, size_t nUsers,
                     size_t nFactors) {
  std::cout << "ALS (native): initializeModel " << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();

  KeyValueDataCollectionPtr initStep1LocalResult =
      initializeStep1Local(rankId, partitionId, nBlocks, nUsers, nFactors);

  /* MPI_Alltoallv to populate initStep2LocalInput */
  ByteBuffer nodeCPs[nBlocks];
  for (size_t i = 0; i < nBlocks; i++) {
    serializeDAALObject((*initStep1LocalResult)[i].get(), nodeCPs[i]);
  }
  KeyValueDataCollectionPtr initStep2LocalInput(new KeyValueDataCollection());
  all2all<NumericTable>(nodeCPs, nBlocks, initStep2LocalInput);

  initializeStep2Local(rankId, partitionId, initStep2LocalInput);

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  std::cout << "ALS (native): initializeModel took " << duration << " secs" << std::endl;
}

training::DistributedPartialResultStep1Ptr computeStep1Local(
    const training::DistributedPartialResultStep4Ptr& partialResultLocal,
    size_t nFactors) {
  /* Create algorithm objects to compute implicit ALS algorithm in the distributed
   * processing mode on the local node using the default method */
  training::Distributed<step1Local> algorithm;
  algorithm.parameter.nFactors = nFactors;

  /* Set input objects for the algorithm */
  algorithm.input.set(training::partialModel,
                      partialResultLocal->get(training::outputOfStep4ForStep1));

  /* Compute partial estimates on local nodes */
  algorithm.compute();

  /* Get the computed partial estimates */
  return algorithm.getPartialResult();
}

NumericTablePtr computeStep2Master(
    const training::DistributedPartialResultStep1Ptr* step1LocalResultsOnMaster,
    size_t nFactors, size_t nBlocks) {
  /* Create algorithm objects to compute implicit ALS algorithm in the distributed
   * processing mode on the master node using the default method */
  training::Distributed<step2Master> algorithm;
  algorithm.parameter.nFactors = nFactors;

  /* Set input objects for the algorithm */
  for (size_t i = 0; i < nBlocks; i++) {
    algorithm.input.add(training::inputOfStep2FromStep1, step1LocalResultsOnMaster[i]);
  }

  /* Compute a partial estimate on the master node from the partial estimates on local
   * nodes */
  algorithm.compute();

  return algorithm.getPartialResult()->get(training::outputOfStep2ForStep4);
}

KeyValueDataCollectionPtr computeStep3Local(
    const NumericTablePtr& offset,
    const training::DistributedPartialResultStep4Ptr& partialResultLocal,
    const KeyValueDataCollectionPtr& step3LocalInput, size_t nFactors) {
  training::Distributed<step3Local> algorithm;
  algorithm.parameter.nFactors = nFactors;

  algorithm.input.set(training::partialModel,
                      partialResultLocal->get(training::outputOfStep4ForStep3));
  algorithm.input.set(training::inputOfStep3FromInit, step3LocalInput);
  algorithm.input.set(training::offset, offset);

  algorithm.compute();

  return algorithm.getPartialResult()->get(training::outputOfStep3ForStep4);
}

training::DistributedPartialResultStep4Ptr computeStep4Local(
    const CSRNumericTablePtr& dataTable, const NumericTablePtr& step2MasterResult,
    const KeyValueDataCollectionPtr& step4LocalInput, size_t nFactors) {
  training::Distributed<step4Local> algorithm;
  algorithm.parameter.nFactors = nFactors;

  algorithm.input.set(training::partialModels, step4LocalInput);
  algorithm.input.set(training::partialData, dataTable);
  algorithm.input.set(training::inputOfStep4FromStep2, step2MasterResult);

  algorithm.compute();

  return algorithm.getPartialResult();
}

void trainModel(size_t rankId, size_t partitionId, size_t nBlocks, size_t nFactors,
                size_t maxIterations) {
  std::cout << "ALS (native): trainModel" << std::endl;

  auto tStart = std::chrono::high_resolution_clock::now();

  training::DistributedPartialResultStep1Ptr step1LocalResultsOnMaster[nBlocks];
  training::DistributedPartialResultStep1Ptr step1LocalResult;
  NumericTablePtr step2MasterResult;
  KeyValueDataCollectionPtr step3LocalResult;
  KeyValueDataCollectionPtr step4LocalInput(new KeyValueDataCollection());

  ByteBuffer nodeCPs[nBlocks];
  ByteBuffer nodeResults;
  ByteBuffer crossProductBuf;
  int crossProductLen;

  for (size_t iteration = 0; iteration < maxIterations; iteration++) {
    auto t1 = std::chrono::high_resolution_clock::now();

    //
    // Update partial users factors
    //
    step1LocalResult = computeStep1Local(itemsPartialResultLocal, nFactors);

    serializeDAALObject(step1LocalResult.get(), nodeResults);

    /* Gathering step1LocalResult on the master */
    gather(rankId, nBlocks, nodeResults, step1LocalResultsOnMaster);

    if (rankId == ccl_root) {
      step2MasterResult =
          computeStep2Master(step1LocalResultsOnMaster, nFactors, nBlocks);
      serializeDAALObject(step2MasterResult.get(), crossProductBuf);
      crossProductLen = crossProductBuf.size();
    }

    ccl_request_t request;

    // MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, ccl_root, MPI_COMM_WORLD);
    ccl_bcast(&crossProductLen, sizeof(int), ccl_dtype_char, ccl_root, NULL, NULL, NULL,
              &request);
    ccl_wait(request);

    if (rankId != ccl_root) {
      crossProductBuf.resize(crossProductLen);
    }
    // MPI_Bcast(&crossProductBuf[0], crossProductLen, MPI_CHAR, ccl_root,
    // MPI_COMM_WORLD);
    ccl_bcast(&crossProductBuf[0], crossProductLen, ccl_dtype_char, ccl_root, NULL, NULL,
              NULL, &request);
    ccl_wait(request);

    step2MasterResult =
        NumericTable::cast(deserializeDAALObject(&crossProductBuf[0], crossProductLen));

    step3LocalResult = computeStep3Local(itemOffset, itemsPartialResultLocal,
                                         itemStep3LocalInput, nFactors);

    /* MPI_Alltoallv to populate step4LocalInput */
    for (size_t i = 0; i < nBlocks; i++) {
      serializeDAALObject((*step3LocalResult)[i].get(), nodeCPs[i]);
    }
    all2all<PartialModel>(nodeCPs, nBlocks, step4LocalInput);

    usersPartialResultLocal = computeStep4Local(transposedDataTable, step2MasterResult,
                                                step4LocalInput, nFactors);

    //
    // Update partial items factors
    //
    step1LocalResult = computeStep1Local(usersPartialResultLocal, nFactors);

    serializeDAALObject(step1LocalResult.get(), nodeResults);

    /* Gathering step1LocalResult on the master */
    gather(rankId, nBlocks, nodeResults, step1LocalResultsOnMaster);

    if (rankId == ccl_root) {
      step2MasterResult =
          computeStep2Master(step1LocalResultsOnMaster, nFactors, nBlocks);
      serializeDAALObject(step2MasterResult.get(), crossProductBuf);
      crossProductLen = crossProductBuf.size();
    }

    // MPI_Bcast(&crossProductLen, sizeof(int), MPI_CHAR, ccl_root, MPI_COMM_WORLD);
    ccl_bcast(&crossProductLen, sizeof(int), ccl_dtype_char, ccl_root, NULL, NULL, NULL,
              &request);
    ccl_wait(request);

    if (rankId != ccl_root) {
      crossProductBuf.resize(crossProductLen);
    }

    // MPI_Bcast(&crossProductBuf[0], crossProductLen, MPI_CHAR, ccl_root,
    // MPI_COMM_WORLD);
    ccl_bcast(&crossProductBuf[0], crossProductLen, ccl_dtype_char, ccl_root, NULL, NULL,
              NULL, &request);
    ccl_wait(request);

    step2MasterResult =
        NumericTable::cast(deserializeDAALObject(&crossProductBuf[0], crossProductLen));

    step3LocalResult = computeStep3Local(userOffset, usersPartialResultLocal,
                                         userStep3LocalInput, nFactors);

    /* MPI_Alltoallv to populate step4LocalInput */
    for (size_t i = 0; i < nBlocks; i++) {
      serializeDAALObject((*step3LocalResult)[i].get(), nodeCPs[i]);
    }
    all2all<PartialModel>(nodeCPs, nBlocks, step4LocalInput);

    itemsPartialResultLocal =
        computeStep4Local(dataTable, step2MasterResult, step4LocalInput, nFactors);

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "ALS (native): iteration " << iteration << " took " << duration
              << " secs" << std::endl;
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  auto durationTotal =
      std::chrono::duration_cast<std::chrono::seconds>(tEnd - tStart).count();
  std::cout << "ALS (native): trainModel took " << durationTotal << " secs" << std::endl;

  /*Gather all itemsPartialResultLocal to itemsPartialResultsMaster on the master and
   * distributing the result over other ranks*/
  // serializeDAALObject(itemsPartialResultLocal.get(), nodeResults);
  // gatherItems(nodeResults, nBlocks);

  // serializeDAALObject(usersPartialResultLocal.get(), nodeResults);
  // gatherUsers(nodeResults, nBlocks);
}

static size_t getOffsetFromOffsetTable(NumericTablePtr offsetTable) {
  size_t ret;
  BlockDescriptor<int> block;
  offsetTable->getBlockOfRows(0, 1, readOnly, block);
  ret = (size_t)((block.getBlockPtr())[0]);
  offsetTable->releaseBlockOfRows(block);

  return ret;
}

/*
 * Class:     org_apache_spark_ml_recommendation_ALSDALImpl
 * Method:    cShuffleData
 * Signature:
 * (Ljava/nio/ByteBuffer;IILorg/apache/spark/ml/recommendation/ALSPartitionInfo;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_org_apache_spark_ml_recommendation_ALSDALImpl_cShuffleData(
    JNIEnv* env, jobject obj, jobject dataBuffer, jint nTotalKeys, jint nBlocks,
    jobject infoObj) {
  //   cout << "cShuffleData: rank " << rankId << endl;
  cout << "RATING_SIZE: " << RATING_SIZE << endl;

  jbyte* ratingsBuf = (jbyte*)env->GetDirectBufferAddress(dataBuffer);

  jlong ratingsNum = env->GetDirectBufferCapacity(dataBuffer) / RATING_SIZE;

  std::vector<RatingPartition> ratingPartitions(nBlocks);

  for (int i = 0; i < ratingsNum; i++) {
    Rating* rating = (Rating*)(ratingsBuf + RATING_SIZE * i);
    int partition = getPartiton(rating->user, nTotalKeys, nBlocks);
    ratingPartitions[partition].push_back(*rating);
  }

  // for (int i = 0; i < nBlocks; i++) {
  //   cout << "Partition " << i << endl;
  //   for (auto r : ratingPartitions[i]) {
  //     cout << r.user << " " << r.item << " " << r.rating << endl;
  //   }
  // }

  size_t newRatingsNum = 0;
  size_t newCsrRowNum = 0;
  Rating* ratings = shuffle_all2all(ratingPartitions, nBlocks, newRatingsNum, newCsrRowNum);

  // Get the class of the input object
  jclass clazz = env->GetObjectClass(infoObj);
  // Get Field references
  jfieldID ratingsNumField = env->GetFieldID(clazz, "ratingsNum", "I");
  jfieldID csrRowNumField = env->GetFieldID(clazz, "csrRowNum", "I");    
  
  env->SetIntField(infoObj, ratingsNumField, newRatingsNum);
  env->SetIntField(infoObj, csrRowNumField, newCsrRowNum);
  
  return env->NewDirectByteBuffer(ratings, newRatingsNum*RATING_SIZE);
}

/*
 * Class:     org_apache_spark_ml_recommendation_ALSDALImpl
 * Method:    cDALImplictALS
 * Signature: (JJIIDDIIILorg/apache/spark/ml/recommendation/ALSResult;)J
 */

JNIEXPORT jlong JNICALL Java_org_apache_spark_ml_recommendation_ALSDALImpl_cDALImplictALS(
    JNIEnv* env, jobject obj, jlong numTableAddr, jlong nUsers, jint nFactors,
    jint maxIter, jdouble regParam, jdouble alpha, jint executor_num, jint executor_cores,
    jint partitionId, jobject resultObj) {
      
  size_t rankId;
  ccl_get_comm_rank(NULL, &rankId);

  dataTable = *((CSRNumericTablePtr*)numTableAddr);
  // dataTable.reset(createFloatSparseTable("/home/xiaochang/github/oneDAL-upstream/samples/daal/cpp/mpi/data/distributed/implicit_als_csr_1.csv"));

  // printNumericTable(dataTable, "cDALImplictALS", 10);
  cout << "ALS (native): Input info: " << endl;
  cout << "- NumberOfRows: " << dataTable->getNumberOfRows() << endl;
  cout << "- NumberOfColumns: " << dataTable->getNumberOfColumns() << endl;
  cout << "- NumberOfRatings: " << dataTable->getDataSize() << endl;
  cout << "- fullNUsers: " << nUsers << endl;

  // Set number of threads for oneDAL to use for each rank
  services::Environment::getInstance()->setNumberOfThreads(executor_cores);
  int nThreadsNew = services::Environment::getInstance()->getNumberOfThreads();
  cout << "oneDAL (native): Number of threads used: " << nThreadsNew << endl;

  int nBlocks = executor_num;
  initializeModel(rankId, partitionId, nBlocks, nUsers, nFactors);
  trainModel(rankId, partitionId, executor_num, nFactors, maxIter);

  auto pUser =
      usersPartialResultLocal->get(training::outputOfStep4ForStep1)->getFactors();
  // auto pUserIndices =
  // usersPartialResultLocal->get(training::outputOfStep4ForStep1)->getIndices();
  auto pItem =
      itemsPartialResultLocal->get(training::outputOfStep4ForStep1)->getFactors();
  // auto pItemIndices =
  // itemsPartialResultsMaster[i]->get(training::outputOfStep4ForStep1)->getIndices();

  std::cout << "\n=== Results for Rank " << rankId << "===\n" << std::endl;
  // std::cout << "Partition ID: " << partitionId << std::endl;
  printNumericTable(pUser, "User Factors (first 10 rows):", 10);
  printNumericTable(pItem, "Item Factors (first 10 rows):", 10);
  std::cout << "User Offset: " << getOffsetFromOffsetTable(userOffset) << std::endl;
  std::cout << "Item Offset: " << getOffsetFromOffsetTable(itemOffset) << std::endl;
  std::cout << std::endl;

  // printNumericTable(userOffset, "userOffset");
  // printNumericTable(itemOffset, "itemOffset");

  // if (rankId == ccl_root) {
  //     for (int i = 0; i < nBlocks; i++) {
  //         printNumericTable(NumericTable::cast((*userOffsetsOnMaster)[i]),
  //         "userOffsetsOnMaster");
  //     }

  //     for (int i = 0; i < nBlocks; i++) {
  //         printNumericTable(NumericTable::cast((*itemOffsetsOnMaster)[i]),
  //         "itemOffsetsOnMaster");
  //     }
  // }

  //    printf("native pUser %ld, pItem %ld", (jlong)&pUser, (jlong)&pItem);

  // Get the class of the input object
  jclass clazz = env->GetObjectClass(resultObj);

  // Fill in rankId
  jfieldID cRankIdField = env->GetFieldID(clazz, "rankId", "J");
  env->SetLongField(resultObj, cRankIdField, (jlong)rankId);

  // Fill in cUsersFactorsNumTab & cItemsFactorsNumTab
  // Get Field references
  jfieldID cUsersFactorsNumTabField = env->GetFieldID(clazz, "cUsersFactorsNumTab", "J");
  jfieldID cItemsFactorsNumTabField = env->GetFieldID(clazz, "cItemsFactorsNumTab", "J");
  // Set factors as result, should use heap memory
  NumericTablePtr* retUser = new NumericTablePtr(pUser);
  NumericTablePtr* retItem = new NumericTablePtr(pItem);
  env->SetLongField(resultObj, cUsersFactorsNumTabField, (jlong)retUser);
  env->SetLongField(resultObj, cItemsFactorsNumTabField, (jlong)retItem);

  // Fill in cUserOffset & cItemOffset
  jfieldID cUserOffsetField = env->GetFieldID(clazz, "cUserOffset", "J");
  assert(cUserOffsetField != NULL);
  env->SetLongField(resultObj, cUserOffsetField,
                    (jlong)getOffsetFromOffsetTable(userOffset));

  jfieldID cItemOffsetField = env->GetFieldID(clazz, "cItemOffset", "J");
  assert(cItemOffsetField != NULL);
  env->SetLongField(resultObj, cItemOffsetField,
                    (jlong)getOffsetFromOffsetTable(itemOffset));

  return 0;
}
