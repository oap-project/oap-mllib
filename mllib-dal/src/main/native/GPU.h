#pragma once

#include <CL/sycl.hpp>
#include <jni.h>
#include <oneapi/ccl.hpp>

int getLocalRank(ccl::communicator &comm, int size, int rank);
std::vector<sycl::device> get_gpus();
// void setGPUContext(ccl::communicator &comm, jint *gpu_idx, int n_gpu);

sycl::device getAssignedGPU(ccl::communicator &comm, int size, int rankId,
                              jint *gpu_indices, int n_gpu);