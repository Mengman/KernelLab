#include "gpulab/testing/random.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    bool ok = true;
    // Compare against the pre-extraction P01/P02 generation order and engine state.
    for (auto seed : {2026u, 20260803u}) {
        std::mt19937 actual_rng(seed), expected_rng(seed);
        for (std::size_t n : {0u, 1u, 33u, 1000u}) {
            std::vector<float> a(n), b(n), c(n);
            gpulab::testing::fill_random_interleaved({a, b, c}, actual_rng);
            std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
            for (std::size_t i = 0; i < n; ++i) {
                for (float value : {a[i], b[i], c[i]}) {
                    ok = (value == distribution(expected_rng)) && ok;
                }
            }
            gpulab::testing::fill_random_interleaved({a, b}, actual_rng);
            for (std::size_t i = 0; i < n; ++i) {
                for (float value : {a[i], b[i]}) {
                    ok = (value == distribution(expected_rng)) && ok;
                }
            }
            ok = (actual_rng == expected_rng) && ok;
        }
    }
    std::mt19937 rng(42);
    std::vector<float> output(100, -1.0f);
    gpulab::testing::fill_random(output, rng, 2.0f, 3.0f);
    for (float value : output) {
        ok = (value >= 2.0f && value < 3.0f) && ok;
    }
    const auto before = rng;
    std::vector<float> a(2, 9.0f), b(1, 9.0f);
    try {
        gpulab::testing::fill_random_interleaved({a, b}, rng);
        ok = false;
    } catch (const std::invalid_argument&) {
        ok = (a == std::vector<float>(2, 9.0f) && b[0] == 9.0f && rng == before) && ok;
    }
    for (auto bounds :
         {std::pair{1.0f, 1.0f},
          std::pair{2.0f, 1.0f},
          std::pair{0.0f, std::numeric_limits<float>::infinity()}}) {
        try {
            gpulab::testing::fill_random(a, rng, bounds.first, bounds.second);
            ok = false;
        } catch (const std::invalid_argument&) {
            ok = (a == std::vector<float>(2, 9.0f) && rng == before) && ok;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << " random input compatibility and contracts\n";
    return ok ? 0 : 1;
}
