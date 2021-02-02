#include <iostream>
#include <chrono>
#include <oneapi/ccl.hpp>

#include "org_apache_spark_ml_util_OneCCL__.h"

// todo: fill initial comm_size and rank_id
size_t comm_size;
size_t rank_id;

std::vector<ccl::communicator> g_comms;

ccl::communicator &getComm() {
    return g_comms[0];
}

JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_c_1init
  (JNIEnv *env, jobject obj, jint size, jint rank, jstring ip_port, jobject param) {
  
  std::cout << "oneCCL (native): init" << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();

  ccl::init();

  const char *str = env->GetStringUTFChars(ip_port, 0);
  ccl::string ccl_ip_port(str);

  auto kvs_attr = ccl::create_kvs_attr();
  kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);

  ccl::shared_ptr_class<ccl::kvs> kvs;
  kvs = ccl::create_main_kvs(kvs_attr);

  g_comms.push_back(ccl::create_communicator(size, rank, kvs));

  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();
  std::cout << "oneCCL (native): init took " << duration << " secs" << std::endl;

  rank_id = getComm().rank();
  comm_size = getComm().size();

  jclass cls = env->GetObjectClass(param);
  jfieldID fid_comm_size = env->GetFieldID(cls, "commSize", "J");
  jfieldID fid_rank_id = env->GetFieldID(cls, "rankId", "J");  

  env->SetLongField(param, fid_comm_size, comm_size);
  env->SetLongField(param, fid_rank_id, rank_id);    
  env->ReleaseStringUTFChars(ip_port, str);

  return 1;
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    c_cleanup
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_c_1cleanup
  (JNIEnv *env, jobject obj) {

  g_comms.pop_back();

  std::cout << "oneCCL (native): cleanup" << std::endl;

}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    isRoot
 * Signature: ()Z
 */
JNIEXPORT jboolean JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_isRoot
  (JNIEnv *env, jobject obj) {    

    return getComm().rank() == 0;
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    rankID
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_rankID
  (JNIEnv *env, jobject obj) {
    return getComm().rank();
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
