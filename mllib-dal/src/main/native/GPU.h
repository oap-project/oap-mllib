#pragma once

#include "service.h"
#include <CL/cl.h>
#include <CL/sycl.hpp>
#include <jni.h>
#include <oneapi/ccl.hpp>

#include "Communicator.hpp"

sycl::queue getAssignedGPU(const ComputeDevice device, jint *gpu_indices);

sycl::queue getQueue(const ComputeDevice device);
preview::spmd::communicator<preview::spmd::device_memory_access::usm>
createDalCommunicator(jint executorNum, jint rank, ccl::string ccl_ip_port);
