#include "gpulab/build_info.h"
#include "gpulab/build_config.h"

namespace gpulab {
std::string_view version() noexcept { return KERNELLAB_VERSION; }
bool cuda_compiled() noexcept { return KERNELLAB_ENABLE_CUDA != 0; }
} // namespace gpulab
