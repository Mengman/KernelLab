#pragma once

#include <initializer_list>
#include <random>
#include <span>

namespace gpulab::testing {

// Fill equally sized arrays in element order: a[0], b[0], a[1], b[1], ... .
// The caller owns the engine and seed. Values use uniform_real_distribution<float>
// in [lower, upper); exact sequences may differ between standard libraries.
// Empty arrays are allowed. Invalid bounds or unequal sizes throw invalid_argument
// before modifying the arrays or consuming the engine.
void fill_random_interleaved(
    std::initializer_list<std::span<float>> arrays,
    std::mt19937& rng,
    float lower = -10.0f,
    float upper = 10.0f);

// Convenience overload for a single array, including vectors through std::span.
void fill_random(
    std::span<float> output,
    std::mt19937& rng,
    float lower = -10.0f,
    float upper = 10.0f);

} // namespace gpulab::testing
