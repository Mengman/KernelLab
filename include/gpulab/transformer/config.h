#pragma once

namespace gpulab::transformer {
struct ModelConfig {
  int vocab_size = 4096;
  int hidden_size = 512;
  int num_layers = 4;
  int num_heads = 8;
  int num_kv_heads = 8;
  int intermediate_size = 1536;
  int max_seq_len = 2048;

  // Throws std::invalid_argument. This is configuration only, not a model implementation.
  void validate() const;
};
} // namespace gpulab::transformer
