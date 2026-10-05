#pragma once
#include <cstddef>

void axpby_host(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n);

void launch_axpby_plain_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n,
    int grid_size,
    int block_size);

void launch_axpby_kernel(
    const float* x,
    const float* y,
    float* o,
    const float alpha,
    const float beta,
    std::size_t n,
    int grid_size,
    int block_size);