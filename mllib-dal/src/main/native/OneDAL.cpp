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

#include <cstring>
#include <daal.h>
#include <iostream>

#include "org_apache_spark_ml_util_OneDAL__.h"
#include "service.h"

using namespace daal;
using namespace daal::data_management;

// Use oneDAL lib function
extern bool daal_check_is_intel_cpu();

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    setNumericTableValue
 * Signature: (JIID)V
 */
JNIEXPORT void JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_setNumericTableValue(
    JNIEnv *, jobject, jlong numTableAddr, jint row, jint column,
    jdouble value) {
    HomogenNumericTable<double> *nt =
        static_cast<HomogenNumericTable<double> *>(
            ((SerializationIfacePtr *)numTableAddr)->get());
    (*nt)[row][column] = (double)value;
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cSetDouble
 * Signature: (JIID)V
 */
JNIEXPORT void JNICALL Java_org_apache_spark_ml_util_OneDAL_00024_cSetDouble
  (JNIEnv *env, jobject, jlong numTableAddr, jint row, jint column, jdouble value) {
  HomogenNumericTable<double> *nt =
          static_cast<HomogenNumericTable<double> *>(
              ((SerializationIfacePtr *)numTableAddr)->get());
  (*nt)[row][column] = (double)value;

}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cSetDoubleBatch
 * Signature: (JI[DII)V
 */
JNIEXPORT void JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_cSetDoubleBatch(
    JNIEnv *env, jobject, jlong numTableAddr, jint curRows, jdoubleArray batch,
    jint numRows, jint numCols) {
    HomogenNumericTable<double> *nt =
        static_cast<HomogenNumericTable<double> *>(
            ((SerializationIfacePtr *)numTableAddr)->get());
    jdouble *values = (jdouble *)env->GetPrimitiveArrayCritical(batch, 0);
    std::memcpy((*nt)[curRows], values, numRows * numCols * sizeof(double));
    env->ReleasePrimitiveArrayCritical(batch, values, JNI_ABORT);
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cAddNumericTable
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_cAddNumericTable(
    JNIEnv *, jobject, jlong rowMergedNumericTableAddr,
    jlong numericTableAddr) {
    data_management::RowMergedNumericTablePtr pRowMergedNumericTable = (*(
        data_management::RowMergedNumericTablePtr *)rowMergedNumericTableAddr);
    data_management::NumericTablePtr pNumericTable =
        (*(data_management::NumericTablePtr *)numericTableAddr);
    pRowMergedNumericTable->addNumericTable(pNumericTable);
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cFreeDataMemory
 * Signature: (J)V
 */
JNIEXPORT void JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_cFreeDataMemory(
    JNIEnv *, jobject, jlong numericTableAddr) {
    data_management::NumericTablePtr pNumericTable =
        (*(data_management::NumericTablePtr *)numericTableAddr);
    pNumericTable->freeDataMemory();
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cCheckPlatformCompatibility
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_cCheckPlatformCompatibility(
    JNIEnv *, jobject) {
    // Only guarantee compatibility and performance on Intel platforms, use
    // oneDAL lib function
    return daal_check_is_intel_cpu();
}

/*
 * Class:     org_apache_spark_ml_util_OneDAL__
 * Method:    cNewCSRNumericTable
 * Signature: ([F[J[JJJ)J
 */
JNIEXPORT jlong JNICALL
Java_org_apache_spark_ml_util_OneDAL_00024_cNewCSRNumericTable(
    JNIEnv *env, jobject, jfloatArray data, jlongArray colIndices,
    jlongArray rowOffsets, jlong nFeatures, jlong nVectors) {

    long numData = env->GetArrayLength(data);

    size_t *resultRowOffsets = NULL;
    size_t *resultColIndices = NULL;
    float *resultData = NULL;

    CSRNumericTable *numericTable = new CSRNumericTable(
        resultData, resultColIndices, resultRowOffsets, nFeatures, nVectors);
    numericTable->allocateDataMemory(numData);
    numericTable->getArrays<float>(&resultData, &resultColIndices,
                                   &resultRowOffsets);

    size_t *pRowOffsets = (size_t *)env->GetLongArrayElements(rowOffsets, 0);
    size_t *pColIndices = (size_t *)env->GetLongArrayElements(colIndices, 0);
    float *pData = env->GetFloatArrayElements(data, 0);

    for (size_t i = 0; i < (size_t)numData; ++i) {
        resultData[i] = pData[i];
        resultColIndices[i] = pColIndices[i];
    }
    for (size_t i = 0; i < (size_t)nVectors + 1; ++i) {
        resultRowOffsets[i] = pRowOffsets[i];
    }

    env->ReleaseLongArrayElements(rowOffsets, (jlong *)pRowOffsets, 0);
    env->ReleaseLongArrayElements(colIndices, (jlong *)pColIndices, 0);
    env->ReleaseFloatArrayElements(data, pData, 0);

    CSRNumericTablePtr *ret = new CSRNumericTablePtr(numericTable);

    return (jlong)ret;
}
