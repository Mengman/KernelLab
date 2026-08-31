#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpulab {
struct DeviceInfo {
  int index{};
  std::string name;
  int compute_major{};
  int compute_minor{};
  int sm_count{};
  int warp_size{};
  int max_threads_per_block{};
  std::size_t total_memory_bytes{};
  std::size_t shared_memory_per_block_bytes{};
};

struct DeviceInventory {
  // disabled: CPU-only build; unavailable: CUDA built but no usable GPU/driver.
  // ready: at least one GPU. Other runtime errors throw, never silently skip.
  std::string status;
  std::string message;
  std::vector<DeviceInfo> devices;
};

[[nodiscard]] DeviceInventory query_devices();
// Framework diagnostic only, not an assignment solution or performance benchmark.
void run_cuda_smoke();
} // namespace gpulab
