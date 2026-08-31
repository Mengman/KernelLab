#include "gpulab/device.h"

#include <stdexcept>

namespace gpulab {
DeviceInventory query_devices() {
  return {"disabled", "CPU-only build; configure a cuda-* preset to enable device inspection.", {}};
}
void run_cuda_smoke() {
  throw std::runtime_error("CUDA backend is not compiled in this build.");
}
} // namespace gpulab
