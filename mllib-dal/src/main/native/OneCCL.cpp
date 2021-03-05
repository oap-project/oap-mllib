#include <iostream>
#include <chrono>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <list>
#include <ifaddrs.h>
#include <netdb.h>

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
  
  std::cerr << "OneCCL (native): init" << std::endl;

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
  std::cerr << "OneCCL (native): init took " << duration << " secs" << std::endl;

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

  std::cerr << "OneCCL (native): cleanup" << std::endl;

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

static const int CCL_IP_LEN = 128;
std::list<std::string> local_host_ips;

static int fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    if (getifaddrs(&ifaddr) < 0) {
        // LOG_ERROR("fill_local_host_ip: can not get host IP");
        return -1;
    }

    const char iface_name[] = "lo";
    local_host_ips.clear();

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL)
            continue;
        if (strstr(ifa->ifa_name, iface_name) == NULL) {
            family = ifa->ifa_addr->sa_family;
            if (family == AF_INET) {
                memset(local_ip, 0, CCL_IP_LEN);
                int res = getnameinfo(
                    ifa->ifa_addr,
                    (family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6),
                    local_ip,
                    CCL_IP_LEN,
                    NULL,
                    0,
                    NI_NUMERICHOST);
                if (res != 0) {
                    std::string s("fill_local_host_ip: getnameinfo error > ");
                    s.append(gai_strerror(res));
                    // LOG_ERROR(s.c_str());
                    return -1;
                }
                local_host_ips.push_back(local_ip);
            }
        }
    }
    if (local_host_ips.empty()) {
        // LOG_ERROR("fill_local_host_ip: can't find interface to get host IP");
        return -1;
    }
    // memset(local_host_ip, 0, CCL_IP_LEN);
    // strncpy(local_host_ip, local_host_ips.front().c_str(), CCL_IP_LEN);

    // for (auto &ip : local_host_ips)
    //   cout << ip << endl;

    freeifaddrs(ifaddr);
    return 0;
}

static bool is_valid_ip(char ip[]) {
  if (fill_local_host_ip() == -1) {
    std::cerr << "fill_local_host_ip error" << std::endl;
  };
  for (std::list<std::string>::iterator it = local_host_ips.begin(); it != local_host_ips.end(); ++it) {
    if (*it == ip) {
      return true;
    }
  }

  return false;
}

/*
 * Class:     org_apache_spark_ml_util_OneCCL__
 * Method:    getAvailPort
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_ml_util_OneCCL_00024_c_1getAvailPort
  (JNIEnv *env, jobject obj, jstring localIP) {

  // start from beginning of dynamic port
  const int port_start_base = 3000;

  char* local_host_ip = (char *) env->GetStringUTFChars(localIP, NULL);

  // check if the input ip is one of host's ips
  if (!is_valid_ip(local_host_ip))
    return -1;

  struct sockaddr_in main_server_address;
  int server_listen_sock;
  in_port_t port = port_start_base;

  if ((server_listen_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    perror("OneCCL (native) getAvailPort error!");
    return -1;
  }

  main_server_address.sin_family = AF_INET;
  main_server_address.sin_addr.s_addr = inet_addr(local_host_ip);
  main_server_address.sin_port = htons(port);

  // search for available port
  while (bind(server_listen_sock,
         (const struct sockaddr *)&main_server_address,
         sizeof(main_server_address)) < 0) {
    port++;
    main_server_address.sin_port = htons(port);
  }

  close(server_listen_sock);  

  env->ReleaseStringUTFChars(localIP, local_host_ip);

  return port;
}
