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

#include "com_intel_oneapi_dal_table_HomogenTableImpl.h"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

typedef shared_ptr<homogen_table> homogenPtr;
typedef shared_ptr<table_metadata> metadataPtr;

mutex mtx;
unordered_map<long, homogenPtr> cHomogenMap;
unordered_map<long, metadataPtr> cMetaMap;

template <typename T> vector<shared_ptr<T>> cVector;

static void saveShareHomogenPtrMap(const long &lg, const homogenPtr &ptr) {
       mtx.lock();
       cHomogenMap[lg] = ptr;
       mtx.unlock();
}

static void saveShareMetaPtrMap(const long &lg, const metadataPtr &ptr) {
       mtx.lock();
       cMetaMap[lg] = ptr;
       mtx.unlock();
}

static void eraseShareHomogenPtrMap(const long &lg) {
       mtx.lock();
       cHomogenMap.erase(lg);
       mtx.unlock();
}

template <typename T>
static void savePtrVector(const shared_ptr<T> &ptr) {
       mtx.lock();
       cVector<T>.push_back(ptr);
       mtx.unlock();
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
 inline jlong MergeHomogenTable(homogen_table &targetTable, homogen_table &sourceTable, const jint cComputeDevice)
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

       std::copy(targetData, targetData+targetDatasize, p.get());
       std::copy(sourceData, sourceData+sourceDatasize, p.get()+targetDatasize);

       const std::vector<sycl::event> dependencies = {};
       homogenPtr resultTablePtr;
       compute_device device = getComputeDevice(cComputeDevice);
       switch(device) {
        case compute_device::host:{
            savePtrVector<T>(p);
            resultTablePtr = std::make_shared<homogen_table>(p.get(), cRowCount, cColCount,
                                                       detail::make_default_delete<const T>(detail::default_host_policy{}),
                                                       targetTable.get_data_layout());
            break;
        }
#ifdef CPU_GPU_PROFILE
        case compute_device::cpu:
        case compute_device::gpu:{
            auto queue = getQueue(device);
            auto data = sycl::malloc_shared<T>(cRowCount * cColCount, queue);
            queue.memcpy(data, p.get(), sizeof(T) * cRowCount * cColCount).wait();
            printf("merge homegentable size =  %ld \n", sizeof(T) * cRowCount * cColCount );
            resultTablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                       detail::make_default_delete<const T>(queue),
                                                       dependencies, targetTable.get_data_layout());
            break;
        }
#endif
        default: {
            return 0;
        }
   }

  jlong add = reinterpret_cast<jlong>(resultTablePtr.get());
  saveShareHomogenPtrMap(add, resultTablePtr);
  return add;
 }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    iInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_iInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jintArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable int init \n");
    jboolean isCopy = true;
    jint *fData = env->GetIntArrayElements(cData, &isCopy);
    const std::vector<sycl::event> dependencies = {};
    homogenPtr tablePtr;
    compute_device device = getComputeDevice(cComputeDevice);
    switch(device) {
       case compute_device::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount,
                                                        cColCount, detail::empty_delete<const int>(),
                                                        getDataLayout(cLayout));
             break;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:
       case compute_device::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<int>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(int) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const int>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
       }
#endif
       default: {
             env->ReleaseIntArrayElements(cData, fData, 0);
             return 0;
       }
    }
    jlong add = reinterpret_cast<jlong>(tablePtr.get());
    saveShareHomogenPtrMap(add, tablePtr);
    env->ReleaseIntArrayElements(cData, fData, 1);
    return add;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    fInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_fInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jfloatArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable float init \n");
    jboolean isCopy = true;
    jfloat *fData = env->GetFloatArrayElements(cData, &isCopy);
    const std::vector<sycl::event> dependencies = {};
    homogenPtr tablePtr;
    compute_device device = getComputeDevice(cComputeDevice);
    switch(device) {
         case compute_device::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const float>(),
                                                        getDataLayout(cLayout));
             break;
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:
         case compute_device::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<float>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(float) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const float>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
#endif
         default: {
             env->ReleaseFloatArrayElements(cData, fData, 0);
             return 0;
         }
    }
    jlong add = reinterpret_cast<jlong>(tablePtr.get());
    saveShareHomogenPtrMap(add, tablePtr);
    env->ReleaseFloatArrayElements(cData, fData, 1);
    return add;
}
/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    dInit
 * Signature: (JJ[DI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_dInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jdoubleArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable double init \n");
    jboolean isCopy = true;
    jdouble *fData = env->GetDoubleArrayElements(cData, &isCopy);
    const std::vector<sycl::event> dependencies = {};
    homogenPtr tablePtr;
    compute_device device = getComputeDevice(cComputeDevice);
    switch(device) {
         case compute_device::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const double>(),
                                                        getDataLayout(cLayout));
             break;
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:
         case compute_device::gpu:{
             printf("getQueue = %ld \n", device);
             auto queue = getQueue(device);
             printf("getQueue = %ld end \n", device);
             auto data = sycl::malloc_shared<double>(cRowCount * cColCount, queue);
             printf("HomegenTable double malloc memory =  %ld \n", sizeof(double) * cRowCount * cColCount );
             queue.memcpy(data, fData, sizeof(double) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const double>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
#endif
         default: {
             env->ReleaseDoubleArrayElements(cData, fData, 0);
             return 0;
         }
    }
    jlong add = reinterpret_cast<jlong>(tablePtr.get());
    saveShareHomogenPtrMap(add, tablePtr);
    env->ReleaseDoubleArrayElements(cData, fData, 1);
    printf("HomogenTable double end \n");
    return add;
}

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    lInit
 * Signature: (JJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_lInit(
    JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jlongArray cData,
    jint cLayout, jint cComputeDevice) {
    printf("HomogenTable long init \n");
    jboolean isCopy = true;
    jlong *fData = env->GetLongArrayElements(cData, &isCopy);
    const std::vector<sycl::event> dependencies = {};
    homogenPtr tablePtr;
    compute_device device = getComputeDevice(cComputeDevice);
    switch(device) {
         case compute_device::host:{
             tablePtr = std::make_shared<homogen_table>(fData, cRowCount, cColCount,
                                                        detail::empty_delete<const long>(),
                                                        getDataLayout(cLayout));
             break;
         }
#ifdef CPU_GPU_PROFILE
         case compute_device::cpu:
         case compute_device::gpu:{
             auto queue = getQueue(device);
             auto data = sycl::malloc_shared<long>(cRowCount * cColCount, queue);
             queue.memcpy(data, fData, sizeof(long) * cRowCount * cColCount).wait();
             tablePtr = std::make_shared<homogen_table>(queue, data, cRowCount, cColCount,
                                                        detail::make_default_delete<const long>(queue),
                                                        dependencies, getDataLayout(cLayout));
             break;
         }
#endif
         default: {
             env->ReleaseLongArrayElements(cData, fData, 0);
             return 0;
         }
    }
    jlong add = reinterpret_cast<jlong>(tablePtr.get());
    saveShareHomogenPtrMap(add, tablePtr);
    env->ReleaseLongArrayElements(cData, fData, 1);
    return add;
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
    metadataPtr metaPtr = std::make_shared<table_metadata>(*mdata);
    jlong add = reinterpret_cast<jlong>(metaPtr.get());
    saveShareMetaPtrMap(add, metaPtr);
    return add;
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
      homogenPtr tablePtr = std::make_shared<homogen_table>();
      jlong add = reinterpret_cast<jlong>(tablePtr.get());
      saveShareHomogenPtrMap(add, tablePtr);
      return add;
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
          long ret = 0;
          if( targetDataType != sourceDataType ) {
             std::cout << "different data type" << std::endl;
             exit(-1);
          } else {
             switch(targetDataType){
                case data_type::int32:{
                     ret = MergeHomogenTable<int>(targetTable, sourceTable, cComputeDevice);
                     break;
                }
                case data_type::int64:{
                     ret = MergeHomogenTable<long>(targetTable, sourceTable, cComputeDevice);
                     break;
                }
                case data_type::float32:{
                     ret = MergeHomogenTable<float>(targetTable, sourceTable, cComputeDevice);
                     break;
                }
                case data_type::float64:{
                     ret = MergeHomogenTable<double>(targetTable, sourceTable, cComputeDevice);
                     break;
                }
                default: {
                    std::cout << "no base type" << std::endl;
                    exit(-1);
                }
             }
          }
          eraseShareHomogenPtrMap(targetTablePtr);
          eraseShareHomogenPtrMap(sourceTablePtr);
          return ret;
       } else if (!targetTable.has_data() && sourceTable.has_data()){
           eraseShareHomogenPtrMap(targetTablePtr);
           return (jlong)sourceTablePtr;
       }else if (targetTable.has_data() && !sourceTable.has_data()){
           eraseShareHomogenPtrMap(sourceTablePtr);
           return (jlong)targetTablePtr;
       }else {
           std::cout << "Both of tables have not data, please check it" << std::endl;
           exit(-1);
       }
 }
