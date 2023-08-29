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

#pragma once

class cclInitSingleton {
public:
    static ccl::shared_ptr_class<ccl::kvs>& get(int size, int rank, string ccl_ip_port) {
        static std::once_flag flag;
        std::call_once(flag, [size, rank, ccl_ip_port] {
            get_instance(size, rank, ccl_ip_port);
        });
        return get_instance(size, rank, ccl_ip_port);
    }

private:
    static ccl::shared_ptr_class<ccl::kvs>& get_instance(int size, int rank, string ccl_ip_port) {
        std::cerr << "OneCCL singleton init" << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();

        ccl::init();

        auto kvs_attr = ccl::create_kvs_attr();
        kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);

        ccl::shared_ptr_class<ccl::kvs> kvs;
        kvs = ccl::create_main_kvs(kvs_attr);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto duration =
            (float)std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cerr << "OneCCL singleton init took " << duration << " secs"
                  << std::endl;
        return kvs;
    }
};