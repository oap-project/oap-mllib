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

#include "com_intel_oneapi_dal_table_ColumnAccessor.h"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;
typedef std::shared_ptr<homogen_table> homogenPtr;


/*
 * Class:     com_intel_oneapi_dal_table_ColumnAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
   jlong cRowEndIndex, jint cComputeDevice) {
  printf("ColumnAccessor PullDouble \n");
  homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
  column_accessor<const double> acc{ htable };
  oneapi::dal::array<double> col_values;
  jdoubleArray newDoubleArray;
  compute_device device = getComputeDevice(cComputeDevice);
  switch(device) {
       case compute_device::host:{
              col_values = acc.pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
              break;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:
       case compute_device::gpu:{
              auto queue = getQueue(device);
              col_values = acc.pull(queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              break;
       }
#endif
       default: {
             return newDoubleArray;
       }
    }
      newDoubleArray = env->NewDoubleArray(col_values.get_count());
      env->SetDoubleArrayRegion(newDoubleArray, 0, col_values.get_count(), col_values.get_data());
      return newDoubleArray;
}

/*
 * Class:     com_intel_oneapi_dal_table_ColumnAccessor
 * Method:    cPull
 * Signature: (JJ)[D
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullFloat
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
   jlong cRowEndIndex, jint cComputeDevice) {
  printf("ColumnAccessor PullFloat \n");
  homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
  column_accessor<const float> acc{ htable };
  oneapi::dal::array<float> col_values;
  jfloatArray newFloatArray;
  compute_device device = getComputeDevice(cComputeDevice);
  switch(device) {
       case compute_device::host:{
              col_values = acc.pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
              break;
       }
#ifdef CPU_GPU_PROFILE
       case compute_device::cpu:
       case compute_device::gpu:{
              auto queue = getQueue(device);
              col_values = acc.pull(queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
              break;
       }
#endif
       default: {
             return newFloatArray;
       }
    }
      newFloatArray = env->NewFloatArray(col_values.get_count());
      env->SetFloatArrayRegion(newFloatArray, 0, col_values.get_count(), col_values.get_data());
      return newFloatArray;
}

/*
* Class:     com_intel_oneapi_dal_table_ColumnAccessor
* Method:    cPull
* Signature: (JJ)[D
*/
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_ColumnAccessor_cPullInt
(JNIEnv *env, jobject, jlong cTableAddr, jlong cColumnIndex, jlong cRowStartIndex,
 jlong cRowEndIndex, jint cComputeDevice) {
    printf("ColumnAccessor PullInt \n");
    homogen_table htable = *reinterpret_cast<homogen_table *>(cTableAddr);
    column_accessor<const int> acc { htable };
    oneapi::dal::array<int> col_values;
    jintArray newIntArray;
    compute_device device = getComputeDevice(cComputeDevice);
switch(device) {
     case compute_device::host:{
            col_values = acc.pull(cColumnIndex, {cRowStartIndex, cRowEndIndex});
            break;
     }
#ifdef CPU_GPU_PROFILE
     case compute_device::cpu:
     case compute_device::gpu:{
            auto queue = getQueue(device);
            col_values = acc.pull(queue, cColumnIndex, {cRowStartIndex, cRowEndIndex});
            break;
     }
#endif
     default: {
           return newIntArray;
     }
   }
    newIntArray = env->NewIntArray(col_values.get_count());
    env->SetIntArrayRegion(newIntArray, 0, col_values.get_count(), col_values.get_data());
    return newIntArray;
}
