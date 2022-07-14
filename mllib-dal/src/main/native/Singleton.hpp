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

namespace oneapi::dal::detail {

namespace v1 {

template <class T>
class singleton {
public:
    static T& get(int size, int rank, ccl::string ip_port) {
        static std::once_flag flag;
        std::call_once(flag, [size, rank, ip_port] {
            get_instance(size, rank, ip_port);
        });
        return get_instance(size, rank, ip_port);
    }

private:
    static T& get_instance(int size, int rank, ccl::string ip_port) {
        static T instance{size, rank, ip_port};
        return instance;
    }
};

} // namespace v1

using v1::singleton;

} // namespace oneapi::dal::detail
