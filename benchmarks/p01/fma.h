#pragma once
#include <cstddef>

void fma_host(const float *a, const float *b, float *c, std::size_t n);

void launch_fma_kernel(const float *a, const float *b, float *c, std::size_t n, int grid_size);