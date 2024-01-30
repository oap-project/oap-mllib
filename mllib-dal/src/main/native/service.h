/* file: service.h */
/*******************************************************************************
 * Copyright 2017-2020 Intel Corporation
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

/*
!  Content:
!    Auxiliary functions used in C++ samples
!******************************************************************************/

#pragma once

#ifdef CPU_GPU_PROFILE
#include <daal_sycl.h>
#else
#include <daal.h>
#endif

using namespace daal::data_management;

#include <algorithm>
#include <cstdarg>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "Logger.h"
#include "error_handling.h"
#ifdef ONEDAL_VERSION_LESS_THAN_OR_EQUAL_TO_2023.2.0
#include "oneapi/dal/table/detail/csr.hpp"
#else
#include "oneapi/dal/table/csr.hpp"
#endif
#include "oneapi/dal/table/homogen.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::detail;

typedef float GpuAlgorithmFPType;  /* Algorithm floating-point type */
typedef double CpuAlgorithmFPType; /* Algorithm floating-point type */
typedef std::vector<daal::byte> ByteBuffer;
typedef std::shared_ptr<homogen_table> HomogenTablePtr;
typedef std::shared_ptr<csr_table> CSRTablePtr;

enum class ComputeDevice { host, cpu, gpu, uninitialized };
const std::string ComputeDeviceString[] = {"HOST", "CPU", "GPU"};

void printNumericTable(const NumericTablePtr &dataTable,
                       const char *message = "", size_t nPrintedRows = 0,
                       size_t nPrintedCols = 0, size_t interval = 10);
size_t serializeDAALObject(SerializationIface *pData, ByteBuffer &buffer);
SerializationIfacePtr deserializeDAALObject(daal::byte *buff, size_t length);
CSRNumericTable *createFloatSparseTable(const std::string &datasetFileName);
ComputeDevice getComputeDeviceByOrdinal(size_t computeDeviceOrdinal);
void saveHomogenTablePtrToVector(const HomogenTablePtr &ptr);
void saveCSRTablePtrToVector(const CSRTablePtr &ptr);

#ifdef CPU_GPU_PROFILE
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

NumericTablePtr homegenToSyclHomogen(NumericTablePtr ntHomogen);
inline void printHomegenTable(const oneapi::dal::table &table) {
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();
    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            logger::print(logger::INFO, "");
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                logger::print(logger::NONE, "%10f",
                              x[i * table.get_column_count() + j]);
            }
            logger::println(logger::NONE, "");
        }
    } else {
        for (std::int64_t i = 0; i < 5; i++) {
            logger::print(logger::INFO, "");
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                logger::print(logger::NONE, "%10f",
                              x[i * table.get_column_count() + j]);
            }
            logger::println(logger::NONE, "");
        }
        logger::println(logger::INFO, "...%ld lines skipped...",
                        (table.get_row_count() - 10));
        for (std::int64_t i = table.get_row_count() - 5;
             i < table.get_row_count(); i++) {
            logger::print(logger::INFO, "");
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                logger::print(logger::NONE, "%10f",
                              x[i * table.get_column_count() + j]);
            }
            logger::println(logger::NONE, "");
        }
    }
    return;
}
#endif
