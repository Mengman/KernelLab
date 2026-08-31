#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gpulab::detail {
inline void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
  }
}
} // namespace gpulab::detail
