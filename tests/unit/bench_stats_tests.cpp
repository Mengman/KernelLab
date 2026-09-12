#include "gpulab/bench/stats.h"

#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
} // namespace

int main() {
  try {
    std::vector<double> samples;
    gpulab::bench::Stats stats = gpulab::bench::summarize(samples);
    require(stats.count == 0, "stats.count miss match");
    require(stats.max == 0.0, "stats.max miss match");
    require(stats.min == 0.0, "stats.min miss match");
    require(stats.median == 0.0, "stats.median miss match");
    require(stats.p90 == 0.0, "stats.p90 miss match");
    require(stats.p99 == 0.0, "stats.p99 miss match");

    samples = {1.0};
    stats = gpulab::bench::summarize(samples);
    require(stats.count == 1, "single value: stats.count miss match");
    require(stats.max == 1.0, "single value: stats.max miss match");
    require(stats.min == 1.0, "single value: stats.min miss match");
    require(stats.median == 1.0, "single value: stats.median miss match");
    require(stats.p90 == 1.0, "single value: stats.p90 miss match");
    require(stats.p99 == 1.0, "single value: stats.p99 miss match");



    samples = {1.0, 1.0, 1.0, 1.0};
    stats = gpulab::bench::summarize(samples);
    require(stats.count == 4, "same value: stats.count miss match");
    require(stats.max == 1.0, "same value: stats.max miss match");
    require(stats.min == 1.0, "same value: stats.min miss match");
    require(stats.median == 1.0, "same value: stats.median miss match");
    require(stats.p90 == 1.0, "same value: stats.p90 miss match");
    require(stats.p99 == 1.0, "same value: stats.p99 miss match");

    samples = {0.0, 7.0, 1.0, 8.0, 3.0, 4.0, 5.0, 6.0, 9.0, 2.0};
    stats = gpulab::bench::summarize(samples);
    require(stats.count == 10, "ten values: stats.count miss match");
    require(stats.max == 9.0, "ten values: stats.max miss match");
    require(stats.min == 0.0, "ten values: stats.min miss match");
    require(stats.median == 4.0, "ten values: stats.median miss match");
    require(stats.p90 == 8.0, "ten values: stats.p90 miss match");
    require(stats.p99 == 9.0, "ten values: stats.p99 miss match");
    std::cout << "Bench stats tests passed.\n";

    } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }

  return 0;
}
