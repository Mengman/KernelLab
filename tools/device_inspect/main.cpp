#include "gpulab/build_info.h"
#include "gpulab/device.h"
#include "gpulab/json.h"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  bool as_json = false;
  bool require_gpu = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    if (arg == "--json") {
      as_json = true;
    } else if (arg == "--require-gpu") {
      require_gpu = true;
    } else if (arg == "--help") {
      std::cout << "Usage: device_inspect [--json] [--require-gpu]\n"
                   "Exit: 0=inspection completed, 1=error, 2=required GPU unavailable.\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << '\n';
      return 1;
    }
  }
  try {
    const auto inventory = gpulab::query_devices();
    if (as_json) {
      std::cout << "{\"schema_version\":1,\"kernellab_version\":"
                << gpulab::json_quote(gpulab::version()) << ",\"cuda_compiled\":"
                << (gpulab::cuda_compiled() ? "true" : "false") << ",\"status\":"
                << gpulab::json_quote(inventory.status) << ",\"message\":"
                << gpulab::json_quote(inventory.message) << ",\"devices\":[";
      bool first = true;
      for (const auto& device : inventory.devices) {
        if (!first) { std::cout << ','; }
        first = false;
        std::cout << "{\"index\":" << device.index << ",\"name\":"
                  << gpulab::json_quote(device.name) << ",\"compute_major\":"
                  << device.compute_major << ",\"compute_minor\":" << device.compute_minor
                  << ",\"sm_count\":" << device.sm_count << ",\"warp_size\":" << device.warp_size
                  << ",\"max_threads_per_block\":" << device.max_threads_per_block
                  << ",\"total_memory_bytes\":" << device.total_memory_bytes
                  << ",\"shared_memory_per_block_bytes\":" << device.shared_memory_per_block_bytes
                  << '}';
      }
      std::cout << "]}\n";
    } else {
      std::cout << "KernelLab " << gpulab::version() << " | " << inventory.status << '\n'
                << inventory.message << '\n';
      for (const auto& device : inventory.devices) {
        std::cout << '[' << device.index << "] " << device.name << " | sm_"
                  << device.compute_major << device.compute_minor << " | SMs=" << device.sm_count
                  << " | memory=" << device.total_memory_bytes << " bytes\n";
      }
    }
    return (require_gpu && inventory.devices.empty()) ? 2 : 0;
  } catch (const std::exception& error) {
    std::cerr << "Device inspection failed: " << error.what() << '\n';
    return 1;
  }
}
