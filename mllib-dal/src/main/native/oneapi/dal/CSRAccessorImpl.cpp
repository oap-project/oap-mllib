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

#include "com_intel_oneapi_dal_table_CSRAccessor.h"
#include "oneapi/dal/table/detail/csr.hpp"
#include "oneapi/dal/table/detail/csr_accessor.hpp"
#include "service.h"

using namespace std;
using namespace oneapi::dal;

/*
 * Class:     com_intel_oneapi_dal_table_CSRAccessor
 * Method:    cPullDouble
 * Signature: (JJJI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_CSRAccessor_cPullDouble
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
   jint computeDeviceOrdinal){
  printf("CSRAccessor PullDouble \n");
  csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
  csr_accessor<const double> acc {htable};
  jdoubleArray newDoubleArray;
  csr_block<double> csr;
  ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
  switch(device) {
         case ComputeDevice::host:{
                csr = acc.pull({cRowStartIndex, cRowEndIndex});
                break;
         }
#ifdef CPU_GPU_PROFILE
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
            std::cout << "CSRTable is not implemented for GPU/CPU device."
                << std::endl;
            exit(-1);
         }
#endif
         default: {
               return newDoubleArray;
         }
      }
      newDoubleArray = env->NewDoubleArray(csr.data.get_count());
      env->SetDoubleArrayRegion(newDoubleArray, 0, csr.data.get_count(), csr.data.get_data());
      return newDoubleArray;
  }

/*
 * Class:     com_intel_oneapi_dal_table_CSRAccessor
 * Method:    cPullFloat
 * Signature: (JJJI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_CSRAccessor_cPullFloat
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
    jint computeDeviceOrdinal){
   printf("CSRAccessor PullFloat \n");
   csr_table htable = *reinterpret_cast<const csr_table *>(cTableAddr);
   csr_accessor<const float> acc { htable };
   jfloatArray newFloatArray;
   csr_block<float> csr;
   ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
   switch(device) {
          case ComputeDevice::host:{
                 csr = acc.pull({cRowStartIndex, cRowEndIndex});
                 break;
          }
 #ifdef CPU_GPU_PROFILE
          case ComputeDevice::cpu:
          case ComputeDevice::gpu:{
            std::cout << "CSRTable is not implemented for GPU/CPU device."
                << std::endl;
            exit(-1);
          }
 #endif
          default: {
                return newFloatArray;
          }
       }
       newFloatArray = env->NewFloatArray(csr.data.get_count());
       env->SetFloatArrayRegion(newFloatArray, 0, csr.data.get_count(), csr.data.get_data());
       return newFloatArray;
   }

/*
 * Class:     com_intel_oneapi_dal_table_CSRAccessor
 * Method:    cPullInt
 * Signature: (JJJI)[I
 */
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_CSRAccessor_cPullInt
  (JNIEnv *env, jobject, jlong cTableAddr, jlong cRowStartIndex, jlong cRowEndIndex,
                   jint computeDeviceOrdinal){
  printf("CSRAccessor PullInt \n");
  csr_table htable = *reinterpret_cast<csr_table *>(cTableAddr);
  csr_accessor<const int> acc { htable };
  jintArray newIntArray;
  csr_block<int> csr;
  ComputeDevice device = getComputeDeviceByOrdinal(computeDeviceOrdinal);
  switch(device) {
         case ComputeDevice::host:{
                csr = acc.pull({cRowStartIndex, cRowEndIndex});
                break;
         }
#ifdef CPU_GPU_PROFILE
         case ComputeDevice::cpu:
         case ComputeDevice::gpu:{
            std::cout << "CSRTable is not implemented for GPU/CPU device."
                << std::endl;
            exit(-1);
         }
#endif
         default: {
               return newIntArray;
         }
      }
      newIntArray = env->NewIntArray(csr.data.get_count());
      env->SetIntArrayRegion(newIntArray, 0, csr.data.get_count(), csr.data.get_data());
      return newIntArray;
  }