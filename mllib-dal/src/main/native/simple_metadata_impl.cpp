#include <cstring>
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <typeinfo>
#include <memory>
#include <string>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "com_intel_oneapi_dal_table_SimpleMetadataImpl.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace std;
using namespace oneapi::dal;


/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetFeatureCount
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetFeatureCount
  (JNIEnv *env, jobject, jlong ctableAddr) {
      printf("SimpleMetadata getfeaturecount \n");
      table_metadata *mdata = ((std::shared_ptr<table_metadata> *)ctableAddr)->get();
      printf("get value : %ld\n", mdata->get_feature_count());
      return (jlong)mdata->get_feature_count();
  }

/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetFeatureType
 * Signature: (JI)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetFeatureType
  (JNIEnv *env, jobject, jlong ctableAddr, jint cindex) {
        printf("SimpleMetadata getfeaturetype \n");
        table_metadata *mdata = ((std::shared_ptr<table_metadata> *)ctableAddr)->get();
        printf("get value : %d\n", mdata->get_feature_type(cindex));
        return (jint)mdata->get_feature_type(cindex);
  }

/*
 * Class:     com_intel_oneapi_dal_table_SimpleMetadataImpl
 * Method:    cGetDataType
 * Signature: (JI)Ljava/lang/String;
 */
JNIEXPORT jint JNICALL Java_com_intel_oneapi_dal_table_SimpleMetadataImpl_cGetDataType
  (JNIEnv *env, jobject, jlong ctableAddr, jint cindex) {
         printf("SimpleMetadata getdatatype \n");
         table_metadata *mdata = ((std::shared_ptr<table_metadata> *)ctableAddr)->get();
         printf("get value : %d\n", mdata->get_data_type(cindex));
         return (jint)mdata->get_data_type(cindex);
  }
