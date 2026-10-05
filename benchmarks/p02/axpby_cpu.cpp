#include "axpby.h"

void axpby_host(const float *x, const float *y, float *o, const float alpha, const float beta,
                std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    o[i] = alpha * x[i] + beta * y[i];
  }
}