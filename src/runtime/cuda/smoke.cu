#include "gpulab/device.h"
#include "check.cuh"

#include <stdexcept>

namespace gpulab {
namespace {
__global__ void write_marker(int* output) { *output = 42; }

struct DeviceMarker {
  int* pointer{};
  DeviceMarker() {
    detail::cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(int)), "cudaMalloc");
  }
  ~DeviceMarker() { (void)cudaFree(pointer); }
  DeviceMarker(const DeviceMarker&) = delete;
  DeviceMarker& operator=(const DeviceMarker&) = delete;
};
} // namespace

void run_cuda_smoke() {
  DeviceMarker allocation;
  write_marker<<<1, 1>>>(allocation.pointer);
  detail::cuda_check(cudaGetLastError(), "write_marker launch");
  detail::cuda_check(cudaDeviceSynchronize(), "write_marker execution");
  int value = 0;
  detail::cuda_check(cudaMemcpy(&value, allocation.pointer, sizeof(value), cudaMemcpyDeviceToHost),
                     "cudaMemcpy marker to host");
  if (value != 42) {
    throw std::runtime_error("CUDA smoke test returned an incorrect marker.");
  }
}
} // namespace gpulab
