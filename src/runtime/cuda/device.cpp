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

  int cuda_driver_version = 0;
  const auto driverResult = cudaDriverGetVersion(&cuda_driver_version);
  if (driverResult != cudaSuccess) {
    return {"unavailable", cudaGetErrorString(driverResult), {}};
  }

  int cuda_runtime_version = 0;
  const auto cudaResult = cudaRuntimeGetVersion(&cuda_runtime_version);
  if (cudaResult != cudaSuccess) {
    return {"unavailable", cudaGetErrorString(cudaResult), {}};
  }

  DeviceInventory inventory{
      "ready", "CUDA devices detected.", cuda_driver_version, cuda_runtime_version, {}};
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
