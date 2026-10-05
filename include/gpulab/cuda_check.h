#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

namespace gpulab {

inline void check_cuda(cudaError_t status, const char *expression, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << expression
              << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace gpulab

#define CUDA_CHECK(call)                                                     \
  do {                                                                      \
    ::gpulab::check_cuda((call), #call, __FILE__, __LINE__);                    \
  } while (false)
