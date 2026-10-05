#include "axpby.h"
#include "gpulab/cuda_check.h"

__global__ void axpby_plain_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n) {

    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < n) {
        o[i] = alpha * x[i] + beta * y[i];
    }
}

void launch_axpby_plain_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n,
    int grid_size,
    int block_size) {
    if (n == 0)
        return;

    axpby_plain_kernel<<<grid_size, block_size>>>(x, y, o, alpha, beta, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void axpby_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n) {

    std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    for (std::size_t i = start; i < n; i += stride) {
        o[i] = alpha * x[i] + beta * y[i];
    }
}

void launch_axpby_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n,
    int grid_size,
    int block_size) {

    if (n == 0)
        return;

    axpby_kernel<<<grid_size, block_size>>>(x, y, o, alpha, beta, n);
    CUDA_CHECK(cudaGetLastError());
}
