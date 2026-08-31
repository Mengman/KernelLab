#include "gpulab/build_config.h"
#include "gpulab/build_info.h"
#include "gpulab/device.h"
#include "gpulab/json.h"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
  if (!condition) { throw std::runtime_error(message); }
}
}

int main() {
  try {
    require(!gpulab::version().empty(), "Missing version");
    require(gpulab::cuda_compiled() == (KERNELLAB_ENABLE_CUDA != 0), "CUDA build flag mismatch");
    require(gpulab::json_quote("") == "\"\"", "Empty JSON string");
    require(gpulab::json_quote("a\n\t\r\"\\") == "\"a\\n\\t\\r\\\"\\\\\"", "JSON escaping");
    require(gpulab::json_quote(std::string_view{"\0", 1}) == "\"\\u0000\"", "JSON NUL escape");
    require(gpulab::json_quote("GB10") == "\"GB10\"", "JSON ordinary string");
    if (!gpulab::cuda_compiled()) {
      const auto inventory = gpulab::query_devices();
      require(inventory.status == "disabled", "CPU backend status");
      require(inventory.devices.empty(), "CPU backend invented devices");
      bool rejected = false;
      try { gpulab::run_cuda_smoke(); } catch (const std::runtime_error&) { rejected = true; }
      require(rejected, "CPU backend must reject GPU execution");
    }
    std::cout << "Runtime tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
