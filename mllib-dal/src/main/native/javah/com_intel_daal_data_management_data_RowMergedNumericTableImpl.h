/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_intel_daal_data_management_data_RowMergedNumericTableImpl */

#ifndef _Included_com_intel_daal_data_management_data_RowMergedNumericTableImpl
#define _Included_com_intel_daal_data_management_data_RowMergedNumericTableImpl
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getDoubleBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getDoubleBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getFloatBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getFloatBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getIntBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getIntBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseDoubleBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseDoubleBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseFloatBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseFloatBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseIntBlockBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseIntBlockBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getDoubleColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getDoubleColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getFloatColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getFloatColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    getIntColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_getIntColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseDoubleColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseDoubleColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseFloatColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseFloatColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    releaseIntColumnBuffer
 * Signature: (JJJJLjava/nio/ByteBuffer;)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_releaseIntColumnBuffer
  (JNIEnv *, jobject, jlong, jlong, jlong, jlong, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    cNewRowMergedNumericTable
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cNewRowMergedNumericTable
  (JNIEnv *, jobject);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    cGetNumberOfColumns
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cGetNumberOfColumns
  (JNIEnv *, jobject, jlong);

/*
 * Class:     com_intel_daal_data_management_data_RowMergedNumericTableImpl
 * Method:    cAddNumericTable
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_data_1management_data_RowMergedNumericTableImpl_cAddNumericTable
  (JNIEnv *, jobject, jlong, jlong);

#ifdef __cplusplus
}
#endif
#endif
