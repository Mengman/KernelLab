#include "gpulab/bench/stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace gpulab::bench {

static std::size_t nearest_rank(double percentile, std::size_t n) {
  const double rank = std::ceil(percentile / 100.0 * static_cast<double>(n));
  const auto index = static_cast<std::size_t>(rank) - 1;
  return std::min(index, n - 1);
}

Stats summarize(std::vector<double> samples) {

  if (samples.empty()) {
    return Stats{};
  }

  Stats stats;
  stats.count = samples.size();
  std::sort(samples.begin(), samples.end());
  stats.min = samples[0];
  stats.max = samples[stats.count - 1];
  stats.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(stats.count);
  stats.median = samples[nearest_rank(50.0, stats.count)];
  stats.p90 = samples[nearest_rank(90.0, stats.count)];
  stats.p99 = samples[nearest_rank(99.0, stats.count)];

  return stats;
}

} // namespace gpulab::bench