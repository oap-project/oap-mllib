/*******************************************************************************
* Copyright 2021 Intel Corporation
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

//cpp ccl host communicator
#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/ccl.hpp"
#include "oneapi/dal/detail/ccl/communicator.hpp"
#include "Singleton.hpp"

namespace de = oneapi::dal::detail;
namespace oneapi::dal::preview::spmd {

namespace backend {
struct ccl {};
} // namespace backend
class ccl_info {
    friend class de::singleton<ccl_info>;

private:
    ccl_info(int size, int rankId, const ccl::string &ipPort) {
        rank = rankId;
        rank_count = size;
        ccl::string ccl_ip_port(ipPort);
        auto kvs_attr = ccl::create_kvs_attr();
        kvs_attr.set<ccl::kvs_attr_id::ip_port>(ccl_ip_port);
        kvs = ccl::create_main_kvs(kvs_attr);
    }

public:
    ccl::shared_ptr_class<ccl::kvs> kvs;
    int rank;
    int rank_count;
};

template <typename Backend>
communicator<device_memory_access::none> make_communicator(int size, int rank, const ccl::string &ipPort) {
    auto& info = de::singleton<ccl_info>::get(size, rank, ipPort);
    // integral cast
    return oneapi::dal::detail::ccl_communicator<device_memory_access::none>{ info.kvs,
                                                                      info.rank,
                                                                      info.rank_count };
}

template <typename Backend>
communicator<device_memory_access::usm> make_communicator(sycl::queue& queue, int size, int rank, const ccl::string &ipPort) {
    auto& info = de::singleton<ccl_info>::get(size, rank, ipPort);
    return oneapi::dal::detail::ccl_communicator<device_memory_access::usm>{
        queue,
        info.kvs,
        oneapi::dal::detail::integral_cast<std::int64_t>(info.rank),
        oneapi::dal::detail::integral_cast<std::int64_t>(info.rank_count)
    };
}

} // namespace oneapi::dal::preview::spmd

#endif
