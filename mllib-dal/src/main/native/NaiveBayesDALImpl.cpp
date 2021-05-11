#include <daal.h>
#include "service.h"
#include "OneCCL.h"
#include "org_apache_spark_ml_classification_NaiveBayesDALImpl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::multinomial_naive_bayes;

/*
 * Class:     org_apache_spark_ml_classification_NaiveBayesDALImpl
 * Method:    cNaiveBayesDALCompute
 * Signature: (JJIIILorg/apache/spark/ml/classification/NaiveBayesResult;)V
 */
JNIEXPORT void JNICALL Java_org_apache_spark_ml_classification_NaiveBayesDALImpl_cNaiveBayesDALCompute
  (JNIEnv *env, jobject obj, jlong featuresTab, jlong labelsTab,
   jint class_num, jint executor_num, jint executor_cores, jobject result) {

}

void trainModel(int nClasses, const ccl::communicator & comm)
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> trainDataSource("1", DataSource::doAllocateNumericTable,
                                                      DataSource::doDictionaryFromContext);
    FileDataSource<CSVFeatureManager> trainLabelsSource("2", DataSource::doAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);

    /* Retrieve the data from input files */
    trainDataSource.loadDataBlock();
    trainLabelsSource.loadDataBlock();

    auto rankId = comm.rank();
    auto nBlocks = comm.size();

    /* Create an algorithm object to train the Naive Bayes model based on the local-node data */
    training::Distributed<step1Local> localAlgorithm(nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(classifier::training::data, trainDataSource.getNumericTable());
    localAlgorithm.input.set(classifier::training::labels, trainLabelsSource.getNumericTable());

    /* Train the Naive Bayes model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called compute() equal number of times */
    if (rankId == ccl_root)
    {
        serializedData.reset(new byte[perNodeArchLength * nBlocks]);
    }

    {
        services::SharedPtr<byte> nodeResults(new byte[perNodeArchLength]);
        dataArch.copyArchiveToArray(nodeResults.get(), perNodeArchLength);

        /* Transfer partial results to step 2 on the root node */
        // MPI_Gather(nodeResults.get(), perNodeArchLength, MPI_CHAR, serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root, MPI_COMM_WORLD);
        ccl::gather(nodeResults.get(), perNodeArchLength, serializedData.get(), perNodeArchLength, comm).wait();
    }

    if (rankId == ccl_root)
    {
        /* Create an algorithm object to build the final Naive Bayes model on the master node */
        training::Distributed<step2Master> masterAlgorithm(nClasses);

        for (size_t i = 0; i < nBlocks; i++)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() + perNodeArchLength * i, perNodeArchLength);

            training::PartialResultPtr dataForStep2FromStep1(new training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local Naive Bayes model as input for the master-node algorithm */
            masterAlgorithm.input.add(training::partialModels, dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the Naive Bayes model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        training::ResultPtr trainingResult = masterAlgorithm.getResult();
    }
}