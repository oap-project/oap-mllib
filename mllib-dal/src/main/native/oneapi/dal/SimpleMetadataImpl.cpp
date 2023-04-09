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

#ifdef CPU_GPU_PROFILE

#include "com_intel_oneapi_dal_table_SimpleMetadataImpl.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace std;
using namespace oneapi::dal;

/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetFeatureCount
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetFeatureCount(
    JNIEnv *env, jobject, jlong cTableAddr) {
    printf("SimpleMetadata getfeaturecount \n");
    table_metadata mdata = *reinterpret_cast<table_metadata *>(cTableAddr);
    return (jlong)mdata.get_feature_count();

}

/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetFeatureType
 * Signature: (JI)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL
Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetFeatureType(
    JNIEnv *env, jobject, jlong cTableAddr, jint cindex) {
    printf("SimpleMetadata getfeaturetype \n");
    table_metadata mdata = *reinterpret_cast<table_metadata *>(cTableAddr);
    return (jint)mdata.get_feature_type(cindex);
}

/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetDataType
 * Signature: (JI)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL
Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetDataType(
    JNIEnv *env, jobject, jlong cTableAddr, jint cindex) {
    printf("SimpleMetadata getdatatype \n");
    table_metadata mdata = *reinterpret_cast<table_metadata *>(cTableAddr);
    return (jint)mdata.get_data_type(cindex);

}
#endif
