#include "gpulab/transformer/config.h"

#include <exception>
#include <iostream>
#include <stdexcept>

using gpulab::transformer::ModelConfig;

namespace {
void expect_invalid(const ModelConfig& config) {
  try { config.validate(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("Invalid model configuration was accepted.");
}
}

int main() {
  try {
    ModelConfig{}.validate();
    for (auto member : {&ModelConfig::vocab_size, &ModelConfig::hidden_size,
                        &ModelConfig::num_layers, &ModelConfig::num_heads,
                        &ModelConfig::num_kv_heads, &ModelConfig::intermediate_size,
                        &ModelConfig::max_seq_len}) {
      for (const int invalid : {0, -1}) {
        ModelConfig config;
        config.*member = invalid;
        expect_invalid(config);
      }
    }
    ModelConfig config;
    config.hidden_size = 513;
    expect_invalid(config);
    config = ModelConfig{};
    config.num_kv_heads = 3;
    expect_invalid(config);
    config = ModelConfig{};
    config.hidden_size = 24; // head_dim=3 cannot form RoPE pairs.
    expect_invalid(config);
    config = ModelConfig{};
    config.num_kv_heads = 2; // Valid GQA.
    config.validate();
    std::cout << "Transformer config tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
