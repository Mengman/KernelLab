#include "gpulab/device.h"
#include "check.cuh"

namespace gpulab {
DeviceInventory query_devices() {
  int count = 0;
  const auto result = cudaGetDeviceCount(&count);
  if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver) {
    return {"unavailable", cudaGetErrorString(result), {}};
  }
  detail::cuda_check(result, "cudaGetDeviceCount");
  if (count == 0) {
    return {"unavailable", "No CUDA devices detected.", {}};
  }
  DeviceInventory inventory{"ready", "CUDA devices detected.", {}};
  for (int index = 0; index < count; ++index) {
    cudaDeviceProp properties{};
    detail::cuda_check(cudaGetDeviceProperties(&properties, index), "cudaGetDeviceProperties");
    inventory.devices.push_back({index, properties.name, properties.major, properties.minor,
                                 properties.multiProcessorCount, properties.warpSize,
                                 properties.maxThreadsPerBlock, properties.totalGlobalMem,
                                 properties.sharedMemPerBlock});
  }
  return inventory;
}
} // namespace gpulab
