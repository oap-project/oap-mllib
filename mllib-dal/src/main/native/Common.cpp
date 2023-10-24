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

#include "Common.hpp"
#include "Logger.h"
#include "error_handling.h"
#include "service.h"

HomogenTablePtr createHomogenTableWithArrayPtr(size_t pNumTabData,
                                               size_t numRows, size_t numClos,
                                               sycl::queue queue) {
    double *htableArray = reinterpret_cast<double *>(pNumTabData);
    auto data = sycl::malloc_shared<double>(numRows * numClos, queue);
    queue.memcpy(data, htableArray, sizeof(double) * numRows * numClos).wait();
    HomogenTablePtr tablePtr = std::make_shared<homogen_table>(
        queue, data, numRows, numClos,
        detail::make_default_delete<const double>(queue));
    return tablePtr;
}
