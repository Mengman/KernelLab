#pragma once

#include <cstddef>
#include <vector>

namespace gpulab::bench {
struct Stats {
  std::size_t count;
  double min = 0.0, max = 0.0, mean = 0.0, median = 0.0, p90 = 0.0, p99 = 0.0;
};

Stats summarize(std::vector<double> samples);

} // namespace gpulab::bench
