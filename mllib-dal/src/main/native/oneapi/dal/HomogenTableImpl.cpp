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
#include <iostream>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <string>
#include <typeinfo>
#include <vector>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oneapi_dal_table_HomogenTableImpl.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace std;
using namespace oneapi::dal;


std::vector<std::shared_ptr<homogen_table>> cVector;

static data_layout getDataLayout(jint cLayout) {
    data_layout layout;
    switch (cLayout) {
    case 0:
        layout = data_layout::unknown;
        break;
    case 1:
        layout = data_layout::row_major;
        break;
    case 2:
        layout = data_layout::column_major;
        break;
    }
    return layout;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    iInit
 * Signature: (JJLjava/nio/ByteBuffer;Ljava/lang/Class;Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_iInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jintArray cData,
    jint cLayout) {
    printf("HomogenTable int init \n");
    jint *fData = env->GetIntArrayElements(cData, NULL);
    #ifdef CPU_GPU_PROFILE
         #ifdef DEFAULT_HOST_POLICY
              homogen_table *h_table = new homogen_table(
                  fData, cRowCount, cColCount, detail::empty_delete<const int>(),
                  getDataLayout(cLayout));
              std::shared_ptr<homogen_table> *tablePtr =
                         new std::shared_ptr<homogen_table>(h_table);
              cVector.push_back(*tablePtr);
              return (jlong)tablePtr;
         #endif
              sycl::queue queue = getQueue();
              homogen_table *h_sycl_table = new homogen_table(queue,
                  fData, cRowCount, cColCount, detail::empty_delete<const int>(),
                  {}, getDataLayout(cLayout));
              std::shared_ptr<homogen_table> *tableSyclPtr =
                      new std::shared_ptr<homogen_table>(h_sycl_table);
              cVector.push_back(*tableSyclPtr);
              return (jlong)tableSyclPtr;

    #endif
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    fInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_fInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jfloatArray cData,
    jint cLayout) {
    printf("HomogenTable float init \n");
    jfloat *fData = env->GetFloatArrayElements(cData, NULL);
    #ifdef CPU_GPU_PROFILE
             #ifdef DEFAULT_HOST_POLICY
                homogen_table *h_table = new homogen_table(
                    fData, cRowCount, cColCount, detail::empty_delete<const float>(),
                    getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tablePtr =
                           new std::shared_ptr<homogen_table>(h_table);
                cVector.push_back(*tablePtr);
                return (jlong)tablePtr;
             #endif
                sycl::queue queue = getQueue();
                homogen_table *h_sycl_table = new homogen_table(queue,
                    fData, cRowCount, cColCount, detail::empty_delete<const float>(),
                    {}, getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tableSyclPtr =
                        new std::shared_ptr<homogen_table>(h_sycl_table);
                cVector.push_back(*tableSyclPtr);
                return (jlong)tableSyclPtr;
    #endif
}
/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    dInit
 * Signature: (JJ[DI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_dInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jdoubleArray cData,
    jint cLayout) {
    printf("HomogenTable double init \n");
    jdouble *fData = env->GetDoubleArrayElements(cData, NULL);
    #ifdef CPU_GPU_PROFILE
           #ifdef DEFAULT_HOST_POLICY
                homogen_table *h_table = new homogen_table(
                    fData, cRowCount, cColCount, detail::empty_delete<const double>(),
                    getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tablePtr =
                           new std::shared_ptr<homogen_table>(h_table);
                cVector.push_back(*tablePtr);
                return (jlong)tablePtr;
           #endif
                sycl::queue queue = getQueue();
                homogen_table *h_sycl_table = new homogen_table(queue,
                    fData, cRowCount, cColCount, detail::empty_delete<const double>(),
                    {}, getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tableSyclPtr =
                        new std::shared_ptr<homogen_table>(h_sycl_table);
                cVector.push_back(*tableSyclPtr);
                return (jlong)tableSyclPtr;
    #endif
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    lInit
 * Signature: (JJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_lInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jlongArray cData,
    jint cLayout) {
    printf("HomogenTable long init \n");
    jlong *fData = env->GetLongArrayElements(cData, NULL);
    #ifdef CPU_GPU_PROFILE
           #ifdef DEFAULT_HOST_POLICY
                homogen_table *h_table = new homogen_table(
                   fData, cRowCount, cColCount, detail::empty_delete<const long>(),
                   getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tablePtr =
                          new std::shared_ptr<homogen_table>(h_table);
                cVector.push_back(*tablePtr);
                return (jlong)tablePtr;

           #endif
                sycl::queue queue = getQueue();
                homogen_table *h_sycl_table = new homogen_table(queue,
                    fData, cRowCount, cColCount, detail::empty_delete<const long>(),
                    {}, getDataLayout(cLayout));
                std::shared_ptr<homogen_table> *tableSyclPtr =
                        new std::shared_ptr<homogen_table>(h_sycl_table);
                cVector.push_back(*tableSyclPtr);
                return (jlong)tableSyclPtr;
    #endif
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetColumnCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetColumnCount(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getcolumncount \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return htable->get_column_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetRowCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetRowCount(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getrowcount \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return htable->get_row_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetKind
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetKind(JNIEnv *env, jobject,
                                                          jlong ctableAddr) {
    printf("HomogenTable getkind \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return htable->get_kind();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDataLayout
 * Signature: ()J
 */
JNIEXPORT jint JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDataLayout(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getDataLayout \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return (jint)htable->get_data_layout();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetMetaData
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetMetaData(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getMetaData \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    table_metadata *mdata = (table_metadata *)&(htable->get_metadata());
    std::shared_ptr<table_metadata> *metadataPtr =
        new std::shared_ptr<table_metadata>(mdata);
    return (jlong)metadataPtr;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetData
 * Signature: ()J
 */
JNIEXPORT jintArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetIntData(JNIEnv *env,
                                                             jobject,
                                                             jlong ctableAddr) {
    printf("HomogenTable getIntData \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    const int *data = htable->get_data<int>();
    const int datasize = htable->get_column_count() * htable->get_row_count();

    jintArray newIntArray = env->NewIntArray(datasize);
    env->SetIntArrayRegion(newIntArray, 0, datasize, data);
    return newIntArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetFloatData
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetFloatData(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getFloatData \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    const float *data = htable->get_data<float>();
    const int datasize = htable->get_column_count() * htable->get_row_count();

    jfloatArray newFloatArray = env->NewFloatArray(datasize);
    env->SetFloatArrayRegion(newFloatArray, 0, datasize, data);
    return newFloatArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetLongData
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetLongData(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getLongData \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    const long *data = htable->get_data<long>();
    const int datasize = htable->get_column_count() * htable->get_row_count();

    jlongArray newLongArray = env->NewLongArray(datasize);
    env->SetLongArrayRegion(newLongArray, 0, datasize, data);
    return newLongArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDoubleData
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDoubleData(
    JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getDoubleData \n");
    homogen_table *htable =
        ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    const double *data = htable->get_data<double>();
    const int datasize = htable->get_column_count() * htable->get_row_count();

    jdoubleArray newDoubleArray = env->NewDoubleArray(datasize);
    env->SetDoubleArrayRegion(newDoubleArray, 0, datasize, data);
    return newDoubleArray;
}

