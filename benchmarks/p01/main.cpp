#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

inline void check_cuda(cudaError_t status, const char *expression,
                       const char *file, int line) {
  if (status != cudaSuccess) {
    std::cerr << file << ':' << line << ": " << expression
              << " failed: " << cudaGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(call) \
  do { \
    check_cuda((call), #call, __FILE__, __LINE__); \
  } while (false)

void fma_host(const float *a, const float *b, float *c, std::size_t n) {
  for (std::size_t i = 0; i < n; i++) {
    c[i] += a[i] * b[i];
  }
}

__global__ void fma_kernel(const float *a, const float *b, float *c, std::size_t n) {
  std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

  for (std::size_t i = start; i < n; i += stride) {
    c[i] += a[i] * b[i];
  }
}

bool run_case(std::size_t n, bool random_input) {
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
    CUDA_CHECK(cudaMemcpy(d_c, initial_c.data(), bytes, cudaMemcpyHostToDevice));
  }

  fma_kernel<<<2, 256>>>(d_a, d_b, d_c, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

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
      std::cerr << "FAIL n=" << n << " random=" << random_input
                << " index=" << i << " CPU=" << expected[i]
                << " GPU=" << actual[i] << " tolerance=" << tolerance << '\n';
      return false;
    }
  }
  std::cout << "PASS n=" << n << " input="
            << (random_input ? "random" : "deterministic") << '\n';
  return true;
}

int main() {
  // Cover empty input, small input, block/grid boundaries, and many loop rounds.
  constexpr std::size_t sizes[] = {0, 1, 2, 31, 32, 33, 255, 256, 257,
                                 511, 512, 513, 1024, 1025, 100003};
  bool all_passed = true;
  for (std::size_t n : sizes) {
    for (bool random_input : {false, true}) {
      if (!run_case(n, random_input)) {
        all_passed = false;
      }
    }
  }
  std::cout << (all_passed ? "All 30 test cases passed." : "Some test cases failed.") << '\n';
  return all_passed ? 0 : 1;
}
