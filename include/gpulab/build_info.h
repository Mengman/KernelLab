#pragma once

#include <string_view>

namespace gpulab {
[[nodiscard]] std::string_view version() noexcept;
[[nodiscard]] bool cuda_compiled() noexcept;
} // namespace gpulab
