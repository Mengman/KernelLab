#include <cstddef>
#include <vector>

#include "gpulab/cuda_check.h"
#include "gpulab/bench/stats.h"

void benchmark_h2d(std::size_t bytes) {
  unsigned char *h_data = nullptr;
  unsigned char *d_data = nullptr;

  CUDA_CHECK(cudaMallocHost(&h_data, bytes));
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  for (std::size_t i = 0; i < bytes; i++) {
    h_data[i] = 1;
  }

  constexpr int warmup = 20;

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, 0));
  }

  CUDA_CHECK(cudaStreamSynchronize(0));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  constexpr int iteration = 100;
  std::vector<double> samples;
  samples.reserve(iteration);

  for (int i = 0; i < iteration; ++i) {
    CUDA_CHECK(cudaEventRecord(start, 0));

    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, 0));

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    samples.push_back(elapsed_ms);
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaFreeHost(h_data));
  CUDA_CHECK(cudaFree(d_data));

  const auto stats = gpulab::bench::summarize(samples);

  std::cout << "Min: " << stats.min << "ms\nMedian: "  << stats.median << "ms" << std::endl;
  float bandwidth = bytes / (stats.median * 1000000);
  std::cout << "H2D effective bandwidth: " << bandwidth << "GB/s" << std::endl;
}

int main() {
  std::cout << "test 1MiB\n";
  benchmark_h2d(1024 * 1024);

  std::cout << "test 16MiB\n";
  benchmark_h2d(1024 * 1024 * 16);

  std::cout << "test 256MiB\n";
  benchmark_h2d(1024 * 1024 * 256);
  return 0;
}
