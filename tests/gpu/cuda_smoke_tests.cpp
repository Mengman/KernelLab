#include "gpulab/device.h"

#include <exception>
#include <iostream>

int main() {
  try {
    const auto inventory = gpulab::query_devices();
    if (inventory.devices.empty()) {
      std::cout << "SKIP: " << inventory.message << '\n';
      return 77;
    }
    gpulab::run_cuda_smoke();
    std::cout << "CUDA allocation, launch, synchronization, and copy succeeded.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
