#pragma once

#include <CL/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <jni.h>

int getLocalRank(ccl::communicator & comm, int size, int rank);
std::vector<sycl::device> get_gpus();
// void setGPUContext(ccl::communicator &comm, jint *gpu_idx, int n_gpu);