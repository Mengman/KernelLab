#include <chrono>
#include <cmath>
#include <vector>

#include "axpby.h"
#include "gpulab/bench/stats.h"
#include "gpulab/cuda_check.h"
#include "gpulab/testing/random.h"

namespace {
using Clock = std::chrono::steady_clock;
constexpr auto kSeed = 20260803u;
constexpr int kBlockSize = 256;
constexpr float eps = 1e-5;
} // namespace

bool benchmark_axpby(std::size_t bytes) {
    constexpr float alpha = 1.5f, beta = -0.5f;
    bool ok = true;
    std::size_t n = bytes / sizeof(float);
    const int gridSize = static_cast<int>((n + kBlockSize - 1) / kBlockSize);

    std::cout << "axpby plain kernel benchmark." << std::endl;

    float *dx, *dy, *dout;
    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));

    std::vector<float> hx(n), hy(n), hout(n);

    std::mt19937 rng(kSeed);
    gpulab::testing::fill_random_interleaved({hx, hy}, rng);
    constexpr int warmup = 20;
    constexpr int iteration = 100;

    // test axpby cpu
    std::cout << "axpby CPU benchmark\n";
    for (int i = 0; i < warmup; ++i) {
        axpby_host(hx.data(), hy.data(), hout.data(), alpha, beta, n);
    }

    std::vector<double> samples_cpu(iteration);

    for (int i = 0; i < iteration; i++) {
        const auto start = Clock::now();
        axpby_host(hx.data(), hy.data(), hout.data(), alpha, beta, n);
        const auto stop = Clock::now();
        samples_cpu[i] = std::chrono::duration<double, std::milli>(stop - start).count();
    }

    const auto stats_cpu = gpulab::bench::summarize(samples_cpu);
    std::cout << "Min: " << stats_cpu.min << "ms\nMedian: " << stats_cpu.median << "ms"
              << std::endl;
    const double bandwidth_cpu = 3 * bytes / (stats_cpu.median * 1e6);
    const double gflops_cpu = 3 * n / (stats_cpu.median * 1e6);
    std::cout << "kernel bandwidth: " << bandwidth_cpu << " GB/s\n";
    std::cout << "kernel operations: " << gflops_cpu << " GFLOP/s\n";
    // CPU benchmark end

    // GPU axpby plain kernel benchmark
    std::cout << "Benchmark axpby_plain_kernel benchmark\n";
    CUDA_CHECK(cudaMemcpyAsync(dx, hx.data(), bytes, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(dy, hy.data(), bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaStreamSynchronize(0));

    for (int i = 0; i < warmup; ++i) {
        launch_axpby_plain_kernel(dx, dy, dout, alpha, beta, n, gridSize, kBlockSize);
    }

    CUDA_CHECK(cudaStreamSynchronize(0));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<double> samples;
    samples.reserve(iteration);

    for (int i = 0; i < iteration; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));

        launch_axpby_plain_kernel(dx, dy, dout, alpha, beta, n, gridSize, kBlockSize);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples.push_back(elapsed_ms);
    }

    std::vector<float> kernelResult(n);

    CUDA_CHECK(cudaMemcpy(kernelResult.data(), dout, bytes, cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < n; i++) {
        if (!std::isfinite(kernelResult[i])
            || std::abs(hout[i] - kernelResult[i]) > eps + eps * std::abs(hout[i])) {
            std::cout << "axpby_plain_kernel result not equal to cpu result" << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        const auto stats = gpulab::bench::summarize(samples);

        std::cout << "Min: " << stats.min << "ms\nMedian: " << stats.median << "ms" << std::endl;

        const double bandwidth = 3 * bytes / (stats.median * 1e6);
        const double gflops = 3 * n / (stats.median * 1e6);

        std::cout << "kernel bandwidth: " << bandwidth << " GB/s\n";
        std::cout << "kernel operations: " << gflops << " GFLOP/s\n";
    }

    // benchmark axpby grid stride kernel
    std::cout << "Benchmark axpby_kernel benchmark\n";
    constexpr int kGridSize = 256;
    CUDA_CHECK(cudaMemset(dout, 0, bytes));

    for (int i = 0; i < warmup; ++i) {
        launch_axpby_kernel(dx, dy, dout, alpha, beta, n, kGridSize, kBlockSize);
    }

    CUDA_CHECK(cudaStreamSynchronize(0));

    std::vector<double> samples_grid_stride(iteration);
    for (int i = 0; i < iteration; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));

        launch_axpby_kernel(dx, dy, dout, alpha, beta, n, kGridSize, kBlockSize);

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        samples_grid_stride[i] = elapsed_ms;
    }

    CUDA_CHECK(cudaMemcpy(kernelResult.data(), dout, bytes, cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < n; i++) {
        if (!std::isfinite(kernelResult[i])
            || std::abs(hout[i] - kernelResult[i]) > eps + eps * std::abs(hout[i])) {
            std::cout << "axpby_kernel result not equal to cpu result" << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        const auto stats_gs = gpulab::bench::summarize(samples_grid_stride);

        std::cout << "Min: " << stats_gs.min << "ms\nMedian: " << stats_gs.median << "ms"
                  << std::endl;

        const double bandwidth_gs = 3 * bytes / (stats_gs.median * 1e6);
        const double gflops_gs = 3 * n / (stats_gs.median * 1e6);

        std::cout << "kernel bandwidth: " << bandwidth_gs << " GB/s\n";
        std::cout << "kernel operations: " << gflops_gs << " GFLOP/s\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dout));

    return ok;
}

int main() {
    bool all_pass = true;
    std::cout << "test 1MiB\n";
    if (!benchmark_axpby(1024 * 1024)) {
        all_pass = false;
    }

    std::cout << "test 16MiB\n";
    if (!benchmark_axpby(16 * 1024 * 1024)) {
        all_pass = false;
    }

    std::cout << "test 256MiB\n";
    if (!benchmark_axpby(256 * 1024 * 1024)) {
        all_pass = false;
    }

    return all_pass ? 0 : 1;
}