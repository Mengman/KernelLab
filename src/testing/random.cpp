#include "gpulab/testing/random.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpulab::testing {

void fill_random_interleaved(
    std::initializer_list<std::span<float>> arrays,
    std::mt19937& rng,
    float lower,
    float upper) {
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower >= upper
        || static_cast<double>(upper) - lower > std::numeric_limits<float>::max()) {
        throw std::invalid_argument("Random bounds must be finite with a positive float range.");
    }
    if (arrays.size() == 0) {
        return;
    }
    const auto size = arrays.begin()->size();
    for (auto array : arrays) {
        if (array.size() != size) {
            throw std::invalid_argument("Interleaved random arrays must have equal sizes.");
        }
    }
    std::uniform_real_distribution<float> distribution(lower, upper);
    for (std::size_t i = 0; i < size; ++i) {
        for (auto array : arrays) {
            array[i] = distribution(rng);
        }
    }
}

void fill_random(std::span<float> output, std::mt19937& rng, float lower, float upper) {
    fill_random_interleaved({output}, rng, lower, upper);
}

} // namespace gpulab::testing
