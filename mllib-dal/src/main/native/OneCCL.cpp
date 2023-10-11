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

#include <chrono>
#include <iostream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ifaddrs.h>
#include <list>
#include <netdb.h>

#include <oneapi/ccl.hpp>

#include "CCLInitSingleton.hpp"
#include "Logger.h"
#include "OneCCL.h"
#include "com_intel_oap_mllib_OneCCL__.h"

extern const size_t ccl_root = 0;

static const int CCL_IP_LEN = 128;
static std::list<std::string> local_host_ips;
static size_t comm_size = 0;
static size_t rank_id = 0;
static std::vector<ccl::communicator> g_comms;
static std::vector<ccl::shared_ptr_class<ccl::kvs>> g_kvs;

ccl::communicator &getComm() { return g_comms[0]; }
ccl::shared_ptr_class<ccl::kvs> &getKvs() { return g_kvs[0]; }

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1init(
    JNIEnv *env, jobject obj, jint size, jint rank, jstring ip_port,
    jobject param) {

    logger::println(logger::INFO, "OneCCL (native): init");

    auto t1 = std::chrono::high_resolution_clock::now();

    ccl::init();

    const char *str = env->GetStringUTFChars(ip_port, 0);
    ccl::string ccl_ip_port(str);

    auto &singletonCCLInit = CCLInitSingleton::get(size, rank, ccl_ip_port);

    g_kvs.push_back(singletonCCLInit.kvs);
    g_comms.push_back(
        ccl::create_communicator(size, rank, singletonCCLInit.kvs));

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
            .count();
    logger::println(logger::INFO, "OneCCL (native): init took %f secs",
                    duration / 1000);

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
 * Class:     com_intel_oap_mllib_OneCCL__
 * Method:    c_init
 * Signature: ()I
 */
JNIEXPORT jint JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_c_1initDpcpp(JNIEnv *env, jobject) {
    logger::printerrln(logger::INFO, "OneCCL (native): init dpcpp");
    ccl::init();

    return 1;
}

JNIEXPORT void JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_c_1cleanup(JNIEnv *env, jobject obj) {
    logger::printerrln(logger::INFO, "OneCCL (native): cleanup");
    g_kvs.pop_back();
    g_comms.pop_back();
}

JNIEXPORT jboolean JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_isRoot(JNIEnv *env, jobject obj) {

    return getComm().rank() == 0;
}

JNIEXPORT jint JNICALL
Java_com_intel_oap_mllib_OneCCL_00024_rankID(JNIEnv *env, jobject obj) {
    return getComm().rank();
}

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_setEnv(
    JNIEnv *env, jobject obj, jstring key, jstring value, jboolean overwrite) {

    char *k = (char *)env->GetStringUTFChars(key, NULL);
    char *v = (char *)env->GetStringUTFChars(value, NULL);

    int err = setenv(k, v, overwrite);

    env->ReleaseStringUTFChars(key, k);
    env->ReleaseStringUTFChars(value, v);

    return err;
}

static int fill_local_host_ip() {
    struct ifaddrs *ifaddr, *ifa;
    int family = AF_UNSPEC;
    char local_ip[CCL_IP_LEN];
    if (getifaddrs(&ifaddr) < 0) {
        logger::printerrln(logger::ERROR,
                           "OneCCL (native): can not get host IP");
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
                    (family == AF_INET) ? sizeof(struct sockaddr_in)
                                        : sizeof(struct sockaddr_in6),
                    local_ip, CCL_IP_LEN, NULL, 0, NI_NUMERICHOST);
                if (res != 0) {
                    std::string s("OneCCL (native): getnameinfo error > ");
                    s.append(gai_strerror(res));
                    logger::printerrln(logger::ERROR, s);
                    return -1;
                }
                local_host_ips.push_back(local_ip);
            }
        }
    }
    if (local_host_ips.empty()) {
        logger::printerrln(
            logger::ERROR,
            "OneCCL (native): can't find interface to get host IP");
        return -1;
    }

    freeifaddrs(ifaddr);

    return 0;
}

static bool is_valid_ip(char ip[]) {
    if (fill_local_host_ip() == -1) {
        logger::printerrln(logger::ERROR,
                           "OneCCL (native): get local host ip error");
        return false;
    };

    for (std::list<std::string>::iterator it = local_host_ips.begin();
         it != local_host_ips.end(); ++it) {
        if (*it == ip) {
            return true;
        }
    }

    return false;
}

JNIEXPORT jint JNICALL Java_com_intel_oap_mllib_OneCCL_00024_c_1getAvailPort(
    JNIEnv *env, jobject obj, jstring localIP) {

    // start from beginning of dynamic port
    const int port_start_base = 3000;

    char *local_host_ip = (char *)env->GetStringUTFChars(localIP, NULL);

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
