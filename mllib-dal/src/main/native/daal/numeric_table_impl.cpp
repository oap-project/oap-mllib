/* file: numeric_table_impl.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "common_helpers_functions.h"

#include "com_intel_daal_data_management_data_NumericTableImpl.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"


using namespace daal::services;
using namespace daal::data_management;


inline static NumericTablePtr * getNIONumericTableObject(JNIEnv * env, jobject thisObj)
{
    // Get a class reference for Java NumericTableFeature
    jclass cls          = env->FindClass("com/intel/daal/data_management/data/NumericTable");
    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    return (NumericTablePtr *)(env->GetLongField(thisObj, objFieldID));
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    allocateDataMemory
 * Signature:()J
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cAllocateDataMemory(JNIEnv * env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj          = env->GetLongField(thisObj, objFieldID);

    NumericTable * tbl = ((NumericTablePtr *)cObj)->get();

    DAAL_CHECK_THROW(((HomogenNumericTable<> *)tbl)->allocateDataMemory());
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    freeDataMemory
 * Signature:()V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cFreeDataMemory(JNIEnv * env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj          = env->GetLongField(thisObj, objFieldID);

    NumericTable * tbl = ((NumericTablePtr *)cObj)->get();

    ((HomogenNumericTable<> *)tbl)->freeDataMemory();
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    getNumberOfColumns
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfColumns(JNIEnv * env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj          = env->GetLongField(thisObj, objFieldID);

    NumericTable * tbl = ((NumericTablePtr *)cObj)->get();

    jlong nColumns = tbl->getNumberOfColumns();

    return nColumns;
}

/*
 * Class:     daal_NumericTableImpl
 * Method:    getNumberOfRows
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfRows(JNIEnv * env, jobject thisObj)
{
    // Get a class reference for Java NumericTable
    jclass cls = env->FindClass("com/intel/daal/data_management/data/NumericTable");

    jfieldID objFieldID = env->GetFieldID(cls, "cObject", "J");
    jlong cObj          = env->GetLongField(thisObj, objFieldID);

    NumericTable * tbl = ((NumericTablePtr *)cObj)->get();

    jlong nRows = tbl->getNumberOfRows();
    return nRows;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNumberOfRows
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNumberOfRows(JNIEnv * env, jobject thisObj, jlong nRow)
{
    NumericTable * table = getNIONumericTableObject(env, thisObj)->get();
    DAAL_CHECK_THROW(table->resize((size_t)nRow));
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNumberOfColumns
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNumberOfColumns(JNIEnv * env, jobject thisObj, jlong nCol)
{
    NumericTable * table = getNIONumericTableObject(env, thisObj)->get();
    DAAL_CHECK_THROW(table->getDictionary()->setNumberOfFeatures((size_t)nCol));
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    isNormalized
 * Signature:(I)Z
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cIsNormalized(JNIEnv * env, jobject thisObj, jint flag)
{
    NumericTable * table = getNIONumericTableObject(env, thisObj)->get();
    jboolean checkResult = (jboolean)table->isNormalized((NumericTableIface::NormalizationType)flag);
    return checkResult;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    setNormalizationFlag
 * Signature:(I)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetNormalizationFlag(JNIEnv * env, jobject thisObj, jint flag)
{
    NumericTable * table = getNIONumericTableObject(env, thisObj)->get();
    jint oldFlag         = (jint)table->setNormalizationFlag((NumericTableIface::NormalizationType)flag);
    return oldFlag;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    getDataLayout
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDataLayout(JNIEnv * env, jobject thisObj, jlong cObject)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getDataLayout();
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    getDataMemoryStatus
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDataMemoryStatus(JNIEnv * env, jobject thisObj, jlong cObject)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getDataMemoryStatus();
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cGetNumberOfCategories
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetNumberOfCategories(JNIEnv * env, jobject thisObj,
                                                                                                          jlong cObject, jint idx)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getNumberOfCategories(idx);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetIntValue
 * Signature: (JJJ)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetIntValue(JNIEnv * env, jobject thisObj, jlong cObject,
                                                                                               jlong column, jlong row)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<int>(column, row);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetFloatValue
 * Signature: (JJJ)F
 */
JNIEXPORT jfloat JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetFloatValue(JNIEnv * env, jobject thisObj, jlong cObject,
                                                                                                   jlong column, jlong row)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<float>(column, row);
}

/*
 * Class:     com_intel_daal_data_management_data_NumericTableImpl
 * Method:    cGetDoubleValue
 * Signature: (JJJ)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetDoubleValue(JNIEnv * env, jobject thisObj, jlong cObject,
                                                                                                     jlong column, jlong row)
{
    NumericTable * table = ((NumericTablePtr *)cObject)->get();
    return table->getValue<double>(column, row);
}


/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cFreeByteBuffer
 * Signature: (Ljava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cFreeByteBuffer(JNIEnv * env, jobject thisObj, jobject byteBuffer)
{
    daal::byte * buffer = (daal::byte *)(env->GetDirectBufferAddress(byteBuffer));
    daal_free(buffer);
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cGetCDataDictionary
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cGetCDataDictionary(JNIEnv * env, jobject thisObj, jlong cTable)
{
    using namespace daal;
    NumericTablePtr * nt             = (NumericTablePtr *)cTable;
    NumericTableDictionaryPtr * dict = new NumericTableDictionaryPtr((*nt)->getDictionarySharedPtr());
    return (jlong)dict;
}

/*
 * Class:     com_intel_daal_data_1management_data_NumericTableImpl
 * Method:    cSetCDataDictionary
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_NumericTableImpl_cSetCDataDictionary(JNIEnv * env, jobject thisObj, jlong cTable,
                                                                                                      jlong cDictionary)
{
    using namespace daal;
    NumericTablePtr * nt             = (NumericTablePtr *)cTable;
    NumericTableDictionaryPtr * dict = (NumericTableDictionaryPtr *)cDictionary;
    (*nt)->setDictionary(dict->get());
}
