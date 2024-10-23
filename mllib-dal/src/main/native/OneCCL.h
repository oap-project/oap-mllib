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

#include <oneapi/ccl.hpp>
#include <vector>

using namespace std;

namespace ccl {
template <class BufferType,
          class = typename std::enable_if<
              is_native_type_supported<BufferType>(), event>::type>
event CCL_API gather(const BufferType *sendbuf, int sendcount,
                     BufferType *recvbuf, int recvcount,
                     const communicator &comm) {
    auto comm_size = comm.size();
    vector<size_t> send_counts(comm_size, 0);
    vector<size_t> recv_counts(comm_size, 0);

    const size_t root_rank = 0;
    send_counts[root_rank] = sendcount;

    if (comm.rank() == root_rank)
        std::fill(recv_counts.begin(), recv_counts.end(), recvcount);

    return ccl::alltoallv(sendbuf, send_counts, recvbuf, recv_counts, comm);
}
} // namespace ccl

ccl::communicator &getComm();
ccl::shared_ptr_class<ccl::kvs> &getKvs();

#ifdef CPU_GPU_PROFILE
#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif
#include "Communicator.hpp"
oneapi::dal::preview::spmd::communicator<
    oneapi::dal::preview::spmd::device_memory_access::usm> &
getDalComm();
#endif
extern const size_t ccl_root;
