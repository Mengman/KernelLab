#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

inline void check_cuda(cudaError_t status, const char *expression, const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << expression
              << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call)                                                                           \
  do {                                                                                             \
    check_cuda((call), #call, __FILE__, __LINE__);                                                 \
  } while (false)
