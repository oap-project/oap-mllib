#include <daal.h>

#include "OneCCL.h"
#include "org_apache_spark_ml_classification_NaiveBayesDALImpl.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::multinomial_naive_bayes;

typedef double algorithmFPType; /* Algorithm floating-point type */

template <training::Method method>
static training::ResultPtr
trainModel(const ccl::communicator &comm, const NumericTablePtr &featuresTab,
           const NumericTablePtr &labelsTab, int nClasses) {
    auto rankId = comm.rank();
    auto nBlocks = comm.size();

    /* Create an algorithm object to train the Naive Bayes model based on the
     * local-node data */
    training::Distributed<step1Local, algorithmFPType, method> localAlgorithm(
        nClasses);

    /* Pass a training data set and dependent values to the algorithm */
    localAlgorithm.input.set(classifier::training::data, featuresTab);
    localAlgorithm.input.set(classifier::training::labels, labelsTab);

    /* Train the Naive Bayes model on local nodes */
    localAlgorithm.compute();

    /* Serialize partial results required by step 2 */
    services::SharedPtr<daal::byte> serializedData;
    InputDataArchive dataArch;
    localAlgorithm.getPartialResult()->serialize(dataArch);
    size_t perNodeArchLength = dataArch.getSizeOfArchive();

    /* Serialized data is of equal size on each node if each node called
     * compute() equal number of times */
    if (rankId == ccl_root) {
        serializedData.reset(new daal::byte[perNodeArchLength * nBlocks]);
    }

    {
        services::SharedPtr<daal::byte> nodeResults(
            new daal::byte[perNodeArchLength]);
        dataArch.copyArchiveToArray(nodeResults.get(), perNodeArchLength);

        /* Transfer partial results to step 2 on the root node */
        // MPI_Gather(nodeResults.get(), perNodeArchLength, MPI_CHAR,
        // serializedData.get(), perNodeArchLength, MPI_CHAR, mpi_root,
        // MPI_COMM_WORLD);
        ccl::gather(nodeResults.get(), perNodeArchLength, serializedData.get(),
                    perNodeArchLength, comm)
            .wait();
    }

    if (rankId == ccl_root) {
        /* Create an algorithm object to build the final Naive Bayes model on
         * the master node */
        training::Distributed<step2Master, algorithmFPType, method>
            masterAlgorithm(nClasses);

        for (size_t i = 0; i < nBlocks; i++) {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(serializedData.get() +
                                           perNodeArchLength * i,
                                       perNodeArchLength);

            training::PartialResultPtr dataForStep2FromStep1(
                new training::PartialResult());
            dataForStep2FromStep1->deserialize(dataArch);

            /* Set the local Naive Bayes model as input for the master-node
             * algorithm */
            masterAlgorithm.input.add(training::partialModels,
                                      dataForStep2FromStep1);
        }

        /* Merge and finalizeCompute the Naive Bayes model on the master node */
        masterAlgorithm.compute();
        masterAlgorithm.finalizeCompute();

        /* Retrieve the algorithm results */
        training::ResultPtr trainingResult = masterAlgorithm.getResult();

        return trainingResult;
    }
    return training::ResultPtr();
}

/*
 * Class:     org_apache_spark_ml_classification_NaiveBayesDALImpl
 * Method:    cNaiveBayesDALCompute
 * Signature: (JJIIILorg/apache/spark/ml/classification/NaiveBayesResult;)V
 */
JNIEXPORT void JNICALL
Java_org_apache_spark_ml_classification_NaiveBayesDALImpl_cNaiveBayesDALCompute(
    JNIEnv *env, jobject obj, jlong pFeaturesTab, jlong pLabelsTab,
    jint class_num, jint executor_num, jint executor_cores, jobject resultObj) {

    ccl::communicator &comm = getComm();

    NumericTablePtr featuresTab = *((NumericTablePtr *)pFeaturesTab);
    NumericTablePtr labelsTab = *((NumericTablePtr *)pLabelsTab);

    // Set number of threads for oneDAL to use for each rank
    services::Environment::getInstance()->setNumberOfThreads(executor_cores);

    int nThreadsNew =
        services::Environment::getInstance()->getNumberOfThreads();
    cout << "oneDAL (native): Number of CPU threads used: " << nThreadsNew
         << endl;


    // Support both dense and csr numeric table
    training::ResultPtr trainingResult;
    if (featuresTab->getDataLayout() == NumericTable::StorageLayout::csrArray) {
        trainingResult = trainModel<training::fastCSR>(comm, featuresTab, labelsTab, class_num);
    } else {
        trainingResult = trainModel<training::defaultDense>(comm, featuresTab, labelsTab, class_num);
    }

    multinomial_naive_bayes::ModelPtr model =
        trainingResult->get(classifier::training::model);
    auto pi = model->getLogP();
    auto theta = model->getLogTheta();

    printNumericTable(pi, "log of class priors");
    printNumericTable(theta, "log of class conditional probabilities");

    // Return all log of class priors (LogP) and log of class conditional
    // probabilities (LogTheta)

    // Get the class of the input object
    jclass clazz = env->GetObjectClass(resultObj);
    // Get Field references
    jfieldID piNumericTableField =
        env->GetFieldID(clazz, "piNumericTable", "J");
    jfieldID thetaNumericTableField =
        env->GetFieldID(clazz, "thetaNumericTable", "J");

    env->SetLongField(resultObj, piNumericTableField, (jlong)pi);
    env->SetLongField(resultObj, thetaNumericTableField, (jlong)theta);
}
