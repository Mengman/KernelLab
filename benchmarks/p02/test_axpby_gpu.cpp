#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "axpby.h"
#include "gpulab/cuda_check.h"
#include "gpulab/testing/random.h"

namespace {
constexpr float kSentinel = 12345.0f;
}

bool run_zero_size_case() {
    bool ok = true;

    std::vector<float> h_x(4, 1.0f), h_y(4, -2.0f), h_guard(4, kSentinel);
    std::size_t bytes = h_x.size() * sizeof(float);

    float *d_x, *d_y, *d_guard;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaMalloc(&d_guard, bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_guard, h_guard.data(), bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaStreamSynchronize(0));
    launch_axpby_plain_kernel(d_x, d_y, d_guard, 1.5f, -0.5f, 0, 1, 32);

    CUDA_CHECK(cudaMemcpyAsync(h_guard.data(), d_guard, bytes, cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    for (std::size_t i = 0; i < h_guard.size(); ++i) {
        if (h_guard[i] != kSentinel) {
            std::cerr << "FAIL case{ n=0 } impl=axpby_gpu_plain index=" << i
                      << " reason=unexpected_write_at_n_zero expected=" << kSentinel
                      << " actual=" << h_guard[i] << std::endl;
            ok = false;
        }
    }

    h_guard = std::vector<float>(4, kSentinel);
    CUDA_CHECK(cudaMemcpyAsync(d_guard, h_guard.data(), bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaStreamSynchronize(0));
    launch_axpby_kernel(d_x, d_y, d_guard, 1.5f, -0.5f, 0, 1, 32);
    CUDA_CHECK(cudaMemcpyAsync(h_guard.data(), d_guard, bytes, cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    for (std::size_t i = 0; i < h_guard.size(); ++i) {
        if (h_guard[i] != kSentinel) {
            std::cerr << "FAIL case{ n=0 } impl=axpby_gpu_grid_stride index=" << i
                      << " reason=unexpected_write_at_n_zero expected=" << kSentinel
                      << " actual=" << h_guard[i] << std::endl;
            ok = false;
        }
    }

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_guard));

    return ok;
}

bool run_empty_pointer_case() {
    launch_axpby_plain_kernel(nullptr, nullptr, nullptr, 1.5f, -0.5f, 0, 1, 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "PASS n=0 nullptr impl=plain\n";

    launch_axpby_kernel(nullptr, nullptr, nullptr, 1.5f, -0.5f, 0, 1, 32);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "PASS n=0 nullptr impl=grid_stride\n";

    return true;
}

namespace {

constexpr auto kSeed = 20260803u;
constexpr int kBlockSize = 32;
using Launch = decltype(&launch_axpby_kernel);

// 两种实现使用相同输入；输出前后各加一个哨兵，检查相邻区域未被写坏。
bool run_nonempty_case(
    const std::vector<float>& x,
    const std::vector<float>& y,
    float alpha,
    float beta,
    bool random_input,
    Launch launch,
    const char* implementation,
    int grid_size) {
    const std::size_t n = x.size();
    const std::size_t bytes = n * sizeof(float);
    std::vector<float> expected(n);
    axpby_host(x.data(), y.data(), expected.data(), alpha, beta, n);
    std::vector<float> actual(n + 2, kSentinel);
    const std::size_t output_bytes = actual.size() * sizeof(float);

    float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, actual.data(), output_bytes, cudaMemcpyHostToDevice));

    bool ok = true;
    std::vector<float> first_pass;
    for (int pass = 0; pass < 2; ++pass) {
        // 第二次不恢复输出，验证覆盖语义，不能累加旧值。
        launch(d_x, d_y, d_out + 1, alpha, beta, n, grid_size, kBlockSize);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(actual.data(), d_out, output_bytes, cudaMemcpyDeviceToHost));
        auto fail = [&](std::size_t index, const char* reason, float reference, float value) {
            std::cerr << "FAIL impl=" << implementation << " n=" << n
                      << " input=" << (random_input ? "random" : "fixed") << " alpha=" << alpha
                      << " beta=" << beta << " grid=" << grid_size << " block=" << kBlockSize
                      << " pass=" << pass << " index=" << index << " reason=" << reason
                      << " expected=" << reference << " actual=" << value << '\n';
            ok = false;
        };
        if (actual.front() != kSentinel) {
            fail(0, "prefix_guard_modified", kSentinel, actual.front());
        }
        if (actual.back() != kSentinel) {
            fail(n + 1, "suffix_guard_modified", kSentinel, actual.back());
        }
        for (std::size_t i = 0; i < n; ++i) {
            const float value = actual[i + 1];
            const float tolerance = 1e-5f + 1e-5f * std::abs(expected[i]);
            if (value == kSentinel) {
                fail(i, "not_written", expected[i], value);
                break;
            }
            if (!std::isfinite(value) || std::abs(value - expected[i]) > tolerance) {
                fail(i, "value_mismatch", expected[i], value);
                break;
            }
            if (pass == 1 && value != first_pass[i + 1]) {
                fail(i, "output_depends_on_old_value", first_pass[i + 1], value);
                break;
            }
        }
        if (pass == 0) {
            first_pass = actual;
        }
    }
    std::vector<float> x_after(n), y_after(n);
    CUDA_CHECK(cudaMemcpy(x_after.data(), d_x, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_after.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    if (x_after != x || y_after != y) {
        std::cerr << "FAIL impl=" << implementation << " n=" << n << " reason=input_modified\n";
        ok = false;
    }
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_out));
    std::cout << (ok ? "PASS" : "FAIL") << " impl=" << implementation << " n=" << n
              << " input=" << (random_input ? "random" : "fixed") << " alpha=" << alpha
              << " beta=" << beta << " grid=" << grid_size << " block=" << kBlockSize << '\n';
    return ok;
}

} // namespace

bool test_axpby_gpu() {
    bool all_passed = true;

    if (!run_zero_size_case()) {
        all_passed = false;
    }

    if (!run_empty_pointer_case()) {
        all_passed = false;
    }

    std::mt19937 rng(kSeed);
    std::size_t passed = 0, failed = 0;
    const std::size_t sizes[] = {1, 3, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1000, 100003};
    const float scalars[][2] = {{1.5f, -0.5f}, {1.0f, 1.0f}, {0.0f, 0.0f}, {-1.0f, 3.0f}};
    for (std::size_t n : sizes) {
        for (const auto& scalar : scalars) {
            for (bool random_input : {false, true}) {
                std::vector<float> x(n), y(n);
                if (random_input) {
                    gpulab::testing::fill_random_interleaved({x, y}, rng);
                } else {
                    for (std::size_t i = 0; i < n; ++i) {
                        x[i] = static_cast<float>(i % 7) * 0.5f - 1.5f;
                        y[i] = static_cast<float>(i % 5) * 0.25f - 0.75f;
                    }
                }
                const int plain_grid = static_cast<int>((n + kBlockSize - 1) / kBlockSize);
                const bool plain_ok = run_nonempty_case(
                    x,
                    y,
                    scalar[0],
                    scalar[1],
                    random_input,
                    launch_axpby_plain_kernel,
                    "plain",
                    plain_grid);
                const bool stride_ok = run_nonempty_case(
                    x,
                    y,
                    scalar[0],
                    scalar[1],
                    random_input,
                    launch_axpby_kernel,
                    "grid_stride",
                    2);
                passed += static_cast<std::size_t>(plain_ok) + static_cast<std::size_t>(stride_ok);
                failed +=
                    static_cast<std::size_t>(!plain_ok) + static_cast<std::size_t>(!stride_ok);
            }
        }
    }
    std::cout << "NONEMPTY_TOTAL=" << passed + failed << " PASSED=" << passed
              << " FAILED=" << failed << " seed=" << kSeed << '\n';
    all_passed = all_passed && failed == 0;
    std::cout << (all_passed ? "All GPU cases passed.\n" : "Some GPU cases failed.\n");
    return all_passed;
}

int main() {
    std::cout << std::unitbuf;
    return test_axpby_gpu() ? 0 : 1;
}
