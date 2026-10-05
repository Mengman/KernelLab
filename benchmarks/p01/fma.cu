#include "gpulab/cuda_check.h"
#include "fma.h"

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

void launch_fma_kernel(const float *a, const float *b, float *c, std::size_t n, int grid_size) {
  fma_kernel<<<grid_size, 256>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());
}
