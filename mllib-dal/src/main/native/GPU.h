#pragma once

#include <oneapi/ccl.hpp>

bool setGPUContext(ccl::communicator &comm, int gpu_idx[], int n_gpu);