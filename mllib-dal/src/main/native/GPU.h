#pragma once

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "service.h"
#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <daal_sycl.h>
#include <jni.h>
#include <oneapi/ccl.hpp>

sycl::queue getAssignedGPU(const ComputeDevice device, ccl::communicator &comm,
                           int size, int rankId, jint *gpu_indices, int n_gpu);

sycl::queue getQueue(const ComputeDevice device);
