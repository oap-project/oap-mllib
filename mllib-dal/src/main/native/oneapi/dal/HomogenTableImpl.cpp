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
#include <memory>
#include <stdio.h>
#include <string.h>
#include <string>
#include <typeinfo>
#include <vector>
#include <mutex>

#ifdef CPU_GPU_PROFILE
#include "Common.hpp"

#include "com_intel_oneapi_dal_table_HomogenTableImpl.h"
#include "service.h"
#include "Logger.h"

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

template <typename T>
 inline jlong MergeHomogenTable(homogen_table &targetTable, homogen_table &sourceTable, const jint computeDeviceOrdinal)
 {
       printf("oneDal merge HomogenTable \n");
       const T *targetData = targetTable.get_data<T>();
       const int targetDatasize = targetTable.get_column_count() * targetTable.get_row_count();

       const T *sourceData = sourceTable.get_data<T>();
       const int sourceDatasize = sourceTable.get_column_count() * sourceTable.get_row_count();

       long cRowCount = targetTable.get_row_count() + sourceTable.get_row_count();
       long cColCount = targetTable.get_column_count();
       std::shared_ptr<T> p(new T[targetDatasize + sourceDatasize], [](T* p){
            delete[] p;
        });
       for (std::int64_t i = 0; i < targetDatasize; i++) {
           p.get()[i] = targetData[i];
       }
       for (std::int64_t i = 0; i < sourceDatasize; i++) {
           p.get()[targetDatasize + i] = sourceData[i];
       }
       const std::vector<sycl::event> dependencies = {};
       HomogenTablePtr resultTablePtr;
       ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
       switch(device) {
        case ComputeDevice::host:{
            staySharePtrToVector<T>(p);
            resultTablePtr = std::make_shared<homogen_table>(p.get(), cRowCount, cColCount,
                                                       detail::make_default_delete<const T>(detail::default_host_policy{}),
                                                       targetTable.get_data_layout());
            break;
        }
        case ComputeDevice::cpu:
        case ComputeDevice::gpu:{
            auto queue = getQueue(device);
            auto data = sycl::malloc_shared<T>(cRowCount * cColCount, queue);
            queue.memcpy(data, p.get(), sizeof(T) * cRowCount * cColCount).wait();
            resultTablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                       detail::make_default_delete<const T>(queue),
                                                       dependencies, targetTable.get_data_layout());
            break;
        }
        default: {
            deviceError();
        }
   }
  saveHomogenTablePtrToVector(resultTablePtr);
  return (jlong)resultTablePtr.get();
 }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    iInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_iInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jintArray cData,
    jint cLayout, jint computeDeviceOrdinal) {
    printf("HomogenTable int init \n");
    jint *fData = static_cast<jint *>(env->GetPrimitiveArrayCritical(cData, NULL));
    if (fData == NULL) {
       std::cout << "Error: unable to obtain critical array" << std::endl;
       exit(-1);
    }
    const std::vector<sycl::event> dependencies = {};
    HomogenTablePtr tablePtr;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch(device) {
       case ComputeDevice::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount,
                                                        cColCount, detail::empty_delete<const int>(),
                                                        getDataLayout(cLayout));
             break;
       }
       case ComputeDevice::cpu:
       case ComputeDevice::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<int>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(int) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const int>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
       }
       default: {
             env->ReleasePrimitiveArrayCritical(cData, fData, 0);
             deviceError();
       }
    }
    saveHomogenTablePtrToVector(tablePtr);
    env->ReleasePrimitiveArrayCritical(cData, fData, 0);
    return (jlong)tablePtr.get();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    fInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_fInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jfloatArray cData,
    jint cLayout, jint computeDeviceOrdinal) {
    printf("HomogenTable float init \n");
    jfloat *fData = static_cast<jfloat *>(env->GetPrimitiveArrayCritical(cData, NULL));
    if (fData == NULL) {
       std::cout << "Error: unable to obtain critical array" << std::endl;
       exit(-1);
    }
    const std::vector<sycl::event> dependencies = {};
    HomogenTablePtr tablePtr;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch(device) {
         case ComputeDevice::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const float>(),
                                                        getDataLayout(cLayout));
             break;
         }
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<float>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(float) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const float>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
         default: {
             env->ReleasePrimitiveArrayCritical(cData, fData, 0);
             deviceError();
         }
    }
    saveHomogenTablePtrToVector(tablePtr);
    env->ReleasePrimitiveArrayCritical(cData, fData, 0);
    return (jlong)tablePtr.get();
}
/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    dInit
 * Signature: (JJ[DI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_dInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jdoubleArray cData,
    jint cLayout, jint computeDeviceOrdinal) {
    printf("HomogenTable double init \n");
    jdouble *fData = static_cast<jdouble *>(env->GetPrimitiveArrayCritical(cData, NULL));
    if (fData == NULL) {
       std::cout << "Error: unable to obtain critical array" << std::endl;
       exit(-1);
    }
    const std::vector<sycl::event> dependencies = {};
    HomogenTablePtr tablePtr;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch(device) {
         case ComputeDevice::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const double>(),
                                                        getDataLayout(cLayout));
             break;
         }
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<double>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(double) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const double>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
         default: {
             env->ReleasePrimitiveArrayCritical(cData, fData, 0);
             deviceError();
         }
    }
     saveHomogenTablePtrToVector(tablePtr);
     env->ReleasePrimitiveArrayCritical(cData, fData, 0);
     return (jlong)tablePtr.get();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    lInit
 * Signature: (JJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_lInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jlongArray cData,
    jint cLayout, jint computeDeviceOrdinal) {
    printf("HomogenTable long init \n");
    jlong *fData = static_cast<jlong *>(env->GetPrimitiveArrayCritical(cData, NULL));
    if (fData == NULL) {
       std::cout << "Error: unable to obtain critical array" << std::endl;
       exit(-1);
    }
    const std::vector<sycl::event> dependencies = {};
    HomogenTablePtr tablePtr;
    ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
    switch(device) {
         case ComputeDevice::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const long>(),
                                                        getDataLayout(cLayout));
             break;
         }
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<long>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(long) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const long>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
         default: {
             env->ReleasePrimitiveArrayCritical(cData, fData, 0);
             deviceError();
         }
    }
     saveHomogenTablePtrToVector(tablePtr);
     env->ReleasePrimitiveArrayCritical(cData, fData, 0);
     return (jlong)tablePtr.get();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetColumnCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetColumnCount(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getcolumncount %ld \n", cTableAddr);
    homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    return htable.get_column_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetRowCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetRowCount(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getrowcount \n");
    homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    return htable.get_row_count();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetKind
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetKind(JNIEnv *env, jobject,
                                                          jlong cTableAddr) {
    printf("HomogenTable getkind \n");
    homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    return htable.get_kind();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDataLayout
 * Signature: ()J
 */
JNIEXPORT jint JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDataLayout(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getDataLayout \n");
    homogen_table  htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    return (jint)htable.get_data_layout();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetMetaData
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetMetaData(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getMetaData \n");
    homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
    const table_metadata *mdata = reinterpret_cast<const table_metadata *>(&htable.get_metadata());
    TableMetadataPtr metaPtr = std::make_shared<table_metadata>(*mdata);
    saveTableMetaPtrToVector(metaPtr);
    return (jlong)metaPtr.get();
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetData
 * Signature: ()J
 */
JNIEXPORT jintArray JNICALL
Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetIntData(JNIEnv *env,
                                                             jobject,
                                                             jlong cTableAddr) {
    printf("HomogenTable getIntData \n");
    homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    const int *data = htable.get_data<int>();
    const int datasize = htable.get_column_count() * htable.get_row_count();
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
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getFloatData \n");
    homogen_table htable = *reinterpret_cast<const homogen_table *>(cTableAddr);
    const float *data = htable.get_data<float>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

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
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getLongData \n");
    homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
    const long *data = htable.get_data<long>();
    const int datasize = htable.get_column_count() * htable.get_row_count();

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
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("HomogenTable getDoubleData \n");
    homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
    const double *data = htable.get_data<double>();
    const int datasize = htable.get_column_count() * htable.get_row_count();
    jdoubleArray newDoubleArray = env->NewDoubleArray(datasize);
    env->SetDoubleArrayRegion(newDoubleArray, 0, datasize, data);
    return newDoubleArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cEmptyTableInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cEmptyTableInit
  (JNIEnv *env, jobject) {
      printf(" init empty HomogenTable \n");
      HomogenTablePtr tablePtr = std::make_shared<homogen_table>();
      saveHomogenTablePtrToVector(tablePtr);
      return (jlong)tablePtr.get();
  }
/*
* Class:     com_intel_oneapi_dal_table_HomogenTableImpl
* Method:    cAddHomogenTable
* Signature: (Ljava/lang/Long;Ljava/lang/Long;)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cAddHomogenTable
 (JNIEnv *env, jobject, jlong targetTablePtr, jlong sourceTablePtr, jint cComputeDevice){
       printf("oneDal addHomogenTable \n");
       homogen_table targetTable = *reinterpret_cast<homogen_table *>(targetTablePtr);
       homogen_table sourceTable = *reinterpret_cast<homogen_table *>(sourceTablePtr);
       const auto targetMetaData = targetTable.get_metadata();
       const auto sourceMetaData = sourceTable.get_metadata();
       if(targetTable.has_data()){
          const auto targetDataType = targetMetaData.get_data_type(0);
          const auto sourceDataType = sourceMetaData.get_data_type(0);
          if( targetDataType != sourceDataType ) {
             std::cout << "different data type" << std::endl;
             exit(-1);
          } else {
             switch(targetDataType){
                case data_type::int32:{
                     return MergeHomogenTable<int>(targetTable, sourceTable, cComputeDevice);
                }
                case data_type::int64:{
                     return MergeHomogenTable<long>(targetTable, sourceTable, cComputeDevice);
                }
                case data_type::float32:{
                     return MergeHomogenTable<float>(targetTable, sourceTable, cComputeDevice);
                }
                case data_type::float64:{
                    return MergeHomogenTable<double>(targetTable, sourceTable, cComputeDevice);
                }
                default: {
                    std::cout << "no base type" << std::endl;
                    exit(-1);
                }
             }
          }
       } else {
           return (jlong)sourceTablePtr;
       }
 }
#endif
