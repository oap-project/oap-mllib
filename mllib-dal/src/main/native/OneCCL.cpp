#include <iostream>
#include <oneapi/ccl.hpp>
#include "org_apache_spark_ml_util_OneCCL__.h"

// todo: fill initial comm_size and rank_id
size_t comm_size;
size_t rank_id;

ccl::communicator *getComm() {
    ccl::shared_ptr_class<ccl::kvs> kvs;
    static ccl::communicator b = ccl::create_communicator(comm_size, rank_id, kvs);
    return &b;
}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_c_1init
  (JNIEnv *env, jobject obj, jobject param) {
  
  std::cout << "oneCCL (native): init" << std::endl;

  ccl::init();

  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  kvs = ccl::create_kvs(main_addr);
   
  auto comm = getComm();

  rank_id = comm->rank();
  comm_size = comm->size();

  jclass cls = env->GetObjectClass(param);
  jfieldID fid_comm_size = env->GetFieldID(cls, "commSize", "J");
  jfieldID fid_rank_id = env->GetFieldID(cls, "rankId", "J");  

  env->SetLongField(param, fid_comm_size, comm_size);
  env->SetLongField(param, fid_rank_id, rank_id);    

  return 1;
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    c_cleanup
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_c_1cleanup
  (JNIEnv *env, jobject obj) {

  std::cout << "oneCCL (native): cleanup" << std::endl;

}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    isRoot
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_isRoot
  (JNIEnv *env, jobject obj) {    

    return getComm()->rank() == 0;
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    rankID
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_rankID
  (JNIEnv *env, jobject obj) {
    return getComm()->rank();
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    setEnv
 * Signature: (Ljava/lang/String;Ljava/lang/String;Z)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_setEnv
  (JNIEnv *env , jobject obj, jstring key, jstring value, jboolean overwrite) {

    char* k = (char *) env->GetStringUTFChars(key, NULL);
    char* v = (char *) env->GetStringUTFChars(value, NULL);

    int err = setenv(k, v, overwrite);

    env->ReleaseStringUTFChars(key, k);
    env->ReleaseStringUTFChars(value, v);

    return err;
}
