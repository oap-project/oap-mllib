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
#include <mutex>

#ifdef CPU_GPU_PROFILE
#include "GPU.h"
#endif

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oneapi_dal_table_CSRTableImpl.h"
#include "oneapi/dal/table/detail/csr.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

typedef std::shared_ptr<table_metadata> TableMetadataPtr;
std::mutex g_mtx;
std::vector<TableMetadataPtr> g_TableMetaDataVector;
template <typename T> std::vector<std::shared_ptr<T>> g_SharedPtrVector;

static void saveTableMetaPtrToVector(const TableMetadataPtr &ptr) {
       g_mtx.lock();
       g_TableMetaDataVector.push_back(ptr);
       g_mtx.unlock();
}

template <typename T>
static void staySharePtrToVector(const std::shared_ptr<T> &ptr) {
       g_mtx.lock();
       g_SharedPtrVector<T>.push_back(ptr);
       g_mtx.unlock();
}


static csr_indexing getCSRIndexing(jint cCSRIndexing) {
    csr_indexing csrIndexing;
    switch (cCSRIndexing) {
    case 0:
        csrIndexing = csr_indexing::one_based;
        break;
    case 1:
        csrIndexing = csr_indexing::zero_based;
        break;
    }
    return csrIndexing;
}

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    iInit
 * Signature: (JJ[I[J[JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_iInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount,
            jintArray cData, jlongArray cColumnIndices, jlongArray cRowIndices,
            jint cCSRIndexing, jint computeDeviceOrdinal) {

  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    fInit
 * Signature: (JJ[F[J[JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_fInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount,
             jfloatArray cData, jlongArray cColumnIndices, jlongArray cRowIndices,
             jint cCSRIndexing, jint computeDeviceOrdinal){


/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    dInit
 * Signature: (JJ[D[J[JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_dInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount,
                jdoubleArray cData, jlongArray cColumnIndices, jlongArray cRowIndices,
                jint cCSRIndexing, jint computeDeviceOrdinal) {
    printf("CSRTable long init \n");
    jboolean isCopy = true;
    jlong *fData = env->GetLongArrayElements(cData, &isCopy);
    jlong *fColumnIndices = env->GetLongArrayElements(cColumnIndices, &isCopy);
    jlong *fRowIndices = env->GetLongArrayElements(cRowIndices, &isCopy);
    const std::vector<sycl::event> dependencies = {};
    CSRTablePtr tablePtr;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch(device) {
         case ComputeDevice::host:{
             tablePtr = std::make_shared<csr_table>(fData,
                                                    cColCount,
                                                    cRowCount,
                                                    fRowIndices,
                                                    fColumnIndices,
                                                    getCSRIndexing(cCSRIndexing));
             break;
         }
#ifdef CPU_GPU_PROFILE
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<long>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(long) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue,
                                                        data,
                                                        cColCount,
                                                        cRowCount,
                                                        fRowIndices,
                                                        fColumnIndices,
                                                        getCSRIndexing(cCSRIndexing));
             break;
         }
#endif
         default: {
             env->ReleaseLongArrayElements(cData, fData, 0);
             return 0;
         }
    }
     saveHomogenTablePtrToVector(tablePtr);
     env->ReleaseLongArrayElements(cData, fData, 1);
     return (jlong)tablePtr.get();
}


}

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    lInit
 * Signature: (JJ[J[J[JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_lInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount,
                jlongArray cData, jlongArray cColumnIndices, jlongArray cRowIndices,
                jint cDataLayout, jint computeDeviceOrdinal){

 }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cEmptyCSRTableInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cEmptyCSRTableInit
  (JNIEnv *env, jobject) {
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetColumnCount
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetColumnCount
  (JNIEnv *env, jobject, jlong cTableAddr){
    printf("CSRTable getcolumncount %ld \n", cTableAddr);
    csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
    return htable.get_column_count();
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetRowCount
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetRowCount
  (JNIEnv *env, jobject, jlong cTableAddr){
    printf("CSRTable getrowcount \n");
    csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
    return htable.get_row_count();
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetKind
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetKind
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getkind \n");
      csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
      return htable.get_kind();
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetDataLayout
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetDataLayout
  (JNIEnv *env, jobject, jlong cTableAddr){
      printf("CSRTable getDataLayout \n");
      csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
      return (jint)htable.get_data_layout();
  }
/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetMetaData
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetMetaData
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getMetaData \n");
      csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
      const table_metadata *mdata = reinterpret_cast<const table_metadata *>(&htable.get_metadata());
      TableMetadataPtr metaPtr = std::make_shared<table_metadata>(*mdata);
      saveTableMetaPtrToVector(metaPtr);
      return (jlong)metaPtr.get();
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetPullCSRBlockIface
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetPullCSRBlockIface
  (JNIEnv *env, jobject, jlong cTableAddr){}

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetIntData
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetIntData
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getIntData \n");
      csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
      const int *data = htable.get_data<int>();
      const int datasize = htable.get_column_count() * htable.get_row_count();
      jintArray newIntArray = env->NewIntArray(datasize);
      env->SetIntArrayRegion(newIntArray, 0, datasize, data);
      return newIntArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetLongData
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetLongData
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getLongData \n");
      csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
      const long *data = htable.get_data<long>();
      const int datasize = htable.get_column_count() * htable.get_row_count();

      jlongArray newLongArray = env->NewLongArray(datasize);
      env->SetLongArrayRegion(newLongArray, 0, datasize, data);
      return newLongArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetFloatData
 * Signature: (J)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetFloatData
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getFloatData \n");
      csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
      const float *data = htable.get_data<float>();
      const int datasize = htable.get_column_count() * htable.get_row_count();

      jfloatArray newFloatArray = env->NewFloatArray(datasize);
      env->SetFloatArrayRegion(newFloatArray, 0, datasize, data);
      return newFloatArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetDoubleData
 * Signature: (J)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetDoubleData
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getDoubleData \n");
      csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
      const double *data = htable.get_data<double>();
      const int datasize = htable.get_column_count() * htable.get_row_count();
      jdoubleArray newDoubleArray = env->NewDoubleArray(datasize);
      env->SetDoubleArrayRegion(newDoubleArray, 0, datasize, data);
      return newDoubleArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetColumnIndices
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetColumnIndices
  (JNIEnv *env, jobject, jlong cTableAddr) {
        printf("CSRTable getColumnIndices \n");
        csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
        const long *column_indices = htable.get_column_indices();

        jlongArray newLongArray = env->NewLongArray(strlen(column_indices));
        env->SetLongArrayRegion(newLongArray, 0, datasize, data);
        return newLongArray;
    }

/*
 * Class:     com_intel_oneapi_dal_table_CSRTableImpl
 * Method:    cGetRowIndices
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_oneapi_dal_table_CSRTableImpl_cGetRowIndices
  (JNIEnv *env, jobject, jlong cTableAddr) {
      printf("CSRTable getRowIndices \n");
      csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
      const long *row_indices = htable.get_row_indices();

      jlongArray newLongArray = env->NewLongArray(strlen(row_indices));
      env->SetLongArrayRegion(newLongArray, 0, datasize, data);
      return newLongArray;
  }