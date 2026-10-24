#include <iostream>
#include <vector>

#include "axpby.h"
#include "gpulab/cuda_check.h"

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

bool test_axpby_gpu() {
    return run_zero_size_case();
}

int main() {
    std::cout << std::unitbuf;
    return test_axpby_gpu() ? 0 : 1;
}