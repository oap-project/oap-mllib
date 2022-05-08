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
#include <queue>
#include <sstream>
#include <string>
#include <vector>

#include "error_handling.h"

typedef double algorithmFpType;
typedef std::vector<daal::byte> ByteBuffer;
enum class compute_device { host, cpu, gpu };

void printNumericTable(const NumericTablePtr &dataTable,
                       const char *message = "", size_t nPrintedRows = 0,
                       size_t nPrintedCols = 0, size_t interval = 10);
size_t serializeDAALObject(SerializationIface *pData, ByteBuffer &buffer);
SerializationIfacePtr deserializeDAALObject(daal::byte *buff, size_t length);
CSRNumericTable *createFloatSparseTable(const std::string &datasetFileName);
compute_device getComputeDevice(size_t cComputeDevice);

#ifdef CPU_GPU_PROFILE
NumericTablePtr homegenToSyclHomogen(NumericTablePtr ntHomogen);
#endif
