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

#include "com_intel_oneapi_dal_table_HomogenTableImpl.h"
#include "oneapi/dal/table/homogen.hpp"

using namespace std;
using namespace oneapi::dal;

static data_layout getDataLayout(jint cLayout){
    data_layout layout;
    switch(cLayout){
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
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_iInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jintArray cData, jint cLayout){
     printf("HomogenTable int init \n");
     jint *fData = env->GetIntArrayElements(cData, NULL);

     homogen_table *h_table = new homogen_table(fData, cRowCount, cColCount, detail::empty_delete<const int>(), getDataLayout(cLayout));
     std::shared_ptr<homogen_table> *tablePtr= new std::shared_ptr<homogen_table>(h_table);
     return (jlong)tablePtr;
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    fInit
 * Signature: (JJ[FI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_fInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jfloatArray cData, jint cLayout) {
       printf("HomogenTable float init \n");
       jfloat *fData = env->GetFloatArrayElements(cData, NULL);
       homogen_table *h_table = new homogen_table(fData, cRowCount, cColCount, detail::empty_delete<const float>(), getDataLayout(cLayout));
       std::shared_ptr<homogen_table> *tablePtr= new std::shared_ptr<homogen_table>(h_table);
       return (jlong)tablePtr;
  }
/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    dInit
 * Signature: (JJ[DI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_dInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jdoubleArray cData, jint cLayout){
      printf("HomogenTable double init \n");
      jdouble *fData = env->GetDoubleArrayElements(cData, NULL);
      homogen_table *h_table = new homogen_table(fData, cRowCount, cColCount, detail::empty_delete<const double>(), getDataLayout(cLayout));
      std::shared_ptr<homogen_table> *tablePtr= new std::shared_ptr<homogen_table>(h_table);
      return (jlong)tablePtr;
    }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    lInit
 * Signature: (JJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_lInit
  (JNIEnv *env, jobject, jlong cRowCount, jlong cColCount, jlongArray cData, jint cLayout){
        printf("HomogenTable long init \n");
        jlong *fData = env->GetLongArrayElements(cData, NULL);
        homogen_table *h_table = new homogen_table(fData, cRowCount, cColCount, detail::empty_delete<const long>(), getDataLayout(cLayout));
        std::shared_ptr<homogen_table> *tablePtr= new std::shared_ptr<homogen_table>(h_table);
        return (jlong)tablePtr;
   }


/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetColumnCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetColumnCount
  (JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getcolumncount \n");
    homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return htable->get_column_count();
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetRowCount
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetRowCount
  (JNIEnv *env, jobject, jlong ctableAddr) {
    printf("HomogenTable getrowcount \n");
    homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
    return htable->get_row_count();
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetKind
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetKind
  (JNIEnv *env, jobject, jlong ctableAddr) {
      printf("HomogenTable getkind \n");
      homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
      return htable->get_kind();
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetDataLayout
 * Signature: ()J
 */
JNIEXPORT jint JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDataLayout
  (JNIEnv *env, jobject, jlong ctableAddr) {
      printf("HomogenTable getDataLayout \n");
      homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
      return (jint)htable->get_data_layout();
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetMetaData
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetMetaData
  (JNIEnv *env, jobject, jlong ctableAddr) {
      printf("HomogenTable getMetaData \n");
      homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
      table_metadata *mdata = (table_metadata *)&(htable->get_metadata());
      std::shared_ptr<table_metadata> *metadataPtr= new std::shared_ptr<table_metadata>(mdata);
      return (jlong)metadataPtr;
  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetPullColumnIface
 * Signature: ()J
 */
//JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetPullColumnIface
//  (JNIEnv *, jobject, jlong tableAddr) {
//  }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetPullCSRBlockIface
 * Signature: ()J
 */
//JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetPullCSRBlockIface
//  (JNIEnv *, jobject, jlong tableAddr);

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetData
 * Signature: ()J
 */
JNIEXPORT jintArray JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetIntData
  (JNIEnv *env, jobject, jlong ctableAddr) {
     printf("HomogenTable getIntData \n");
     homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
     const int *data = htable->get_data<int>();
     const int datasize = htable->get_column_count() * htable->get_row_count();;
     jintArray newIntArray = env->NewIntArray(datasize);
     env->SetIntArrayRegion(newIntArray, 0, datasize, data);
     return newIntArray;
  }

  /*
   * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
   * Method:    cGetFloatData
   * Signature: (J)[F
   */
  JNIEXPORT jfloatArray JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetFloatData
    (JNIEnv *env, jobject, jlong ctableAddr) {
     printf("HomogenTable getFloatData \n");
     homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
     const float *data = htable->get_data<float>();
     const int datasize = htable->get_column_count() * htable->get_row_count();;
     jfloatArray newFloatArray = env->NewFloatArray(datasize);
     env->SetFloatArrayRegion(newFloatArray, 0, datasize, data);
     return newFloatArray;
    }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetLongData
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetLongData
  (JNIEnv *env, jobject, jlong ctableAddr){
       printf("HomogenTable getLongData \n");
       homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
       const long *data = htable->get_data<long>();
       const int datasize = htable->get_column_count() * htable->get_row_count();;
       jlongArray newLongArray = env->NewLongArray(datasize);
       env->SetLongArrayRegion(newLongArray, 0, datasize, data);
       return newLongArray;
  }

  /*
   * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
   * Method:    cGetDoubleData
   * Signature: (J)[D
   */
  JNIEXPORT jdoubleArray JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetDoubleData
    (JNIEnv *env, jobject, jlong ctableAddr) {
           printf("HomogenTable getDoubleData \n");
           homogen_table *htable = ((std::shared_ptr<homogen_table> *)ctableAddr)->get();
           const double *data = htable->get_data<double>();
           const int datasize = htable->get_column_count() * htable->get_row_count();;
           jdoubleArray newDoubleArray = env->NewDoubleArray(datasize);
           env->SetDoubleArrayRegion(newDoubleArray, 0, datasize, data);
           return newDoubleArray;
    }

/*
 * Class:     com_intel_oneapi_dal_table_HomogenTableImpl
 * Method:    cGetAccessIfacehost
 * Signature: ()J
 */
//JNIEXPORT jlong JNICALL Java_com_intel_oneapi_dal_table_HomogenTableImpl_cGetAccessIfacehost
//  (JNIEnv *, jobject, jlong tableAddr);
