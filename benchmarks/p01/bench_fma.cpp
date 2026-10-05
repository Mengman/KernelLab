#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "gpulab/cuda_check.h"
#include "fma.h"
#include "gpulab/bench/stats.h"

bool benchmark_fma(std::size_t n, bool random_input, int grid_size) {
  std::vector<float> a(n), b(n), initial_c(n);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = random_input ? distribution(rng) : static_cast<float>(i % 17);
    b[i] = random_input ? distribution(rng) : static_cast<float>(i % 13) + 1.0f;
    initial_c[i] = random_input ? distribution(rng) : static_cast<float>(i % 7) + 0.5f;
  }
  auto expected = initial_c;
  fma_host(a.data(), b.data(), expected.data(), n);

  // Allocate one element for n == 0 so the kernel itself is tested without
  // relying on zero-byte allocation behavior.
  const std::size_t bytes = n * sizeof(float);
  const std::size_t allocation_bytes = n == 0 ? sizeof(float) : bytes;
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

  CUDA_CHECK(cudaMalloc(&d_a, allocation_bytes));
  CUDA_CHECK(cudaMalloc(&d_b, allocation_bytes));
  CUDA_CHECK(cudaMalloc(&d_c, allocation_bytes));
  if (n != 0) {
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));
  }

  constexpr int warmup = 20;
  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemcpy(d_c, initial_c.data(), bytes, cudaMemcpyHostToDevice));
    launch_fma_kernel(d_a, d_b, d_c, n, grid_size);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int iteration = 100;
  std::vector<double> samples;
  samples.reserve(iteration);

  for (int i = 0; i < iteration; ++i) {
    CUDA_CHECK(cudaMemcpy(d_c, initial_c.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start, 0));

    launch_fma_kernel(d_a, d_b, d_c, n, grid_size);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back(elapsed_ms);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> actual(n);
  if (n != 0) {
    CUDA_CHECK(cudaMemcpy(actual.data(), d_c, bytes, cudaMemcpyDeviceToHost));
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  constexpr float absolute_tolerance = 1e-5f;
  constexpr float relative_tolerance = 1e-5f;
  for (std::size_t i = 0; i < n; ++i) {
    const float tolerance = absolute_tolerance + relative_tolerance * std::abs(expected[i]);
    if (std::abs(expected[i] - actual[i]) > tolerance) {
      std::cerr << "FAIL n=" << n << " random=" << random_input << " index=" << i
                << " CPU=" << expected[i] << " GPU=" << actual[i] << " tolerance=" << tolerance
                << '\n';
      return false;
    }
  }

  const auto stats = gpulab::bench::summarize(samples);

  float gflop = 2.0 * n / (stats.median * 1000000);
  std::cout << " FMA =" << gflop << "GFLOP/s" << std::endl;
  return true;
}

int main() {
  constexpr std::size_t mb = 1024 * 1024;
  bool passed = true;
  std::cout << "test 1MiB\n";
  int grid_sizes[] = {2, 32, 128, 256, 512};
  for (auto gs : grid_sizes) {
    std::cout << "grid_size=" << gs;
    if (!benchmark_fma(mb / sizeof(float), true, gs)) {
      passed = false;
    }
  }

  std::cout << "\ntest 16MiB\n";
  for (auto gs : grid_sizes) {
    std::cout << "grid_size=" << gs;
    if (!benchmark_fma((16 * mb) / sizeof(float), true, gs)) {
      passed = false;
    }
  }

  std::cout << "\ntest 256MiB\n";
  for (auto gs : grid_sizes) {
    std::cout << "grid_size=" << gs;
    if (!benchmark_fma((256 * mb) / sizeof(float), true, gs)) {
      passed = false;
    }
  }

  return passed ? 0 : 1;
}
