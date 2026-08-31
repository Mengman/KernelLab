#include "gpulab/transformer/config.h"

#include <stdexcept>

namespace gpulab::transformer {
void ModelConfig::validate() const {
  if (vocab_size <= 0 || hidden_size <= 0 || num_layers <= 0 || num_heads <= 0 ||
      num_kv_heads <= 0 || intermediate_size <= 0 || max_seq_len <= 0) {
    throw std::invalid_argument("All model dimensions must be positive.");
  }
  if (hidden_size % num_heads != 0) {
    throw std::invalid_argument("hidden_size must be divisible by num_heads.");
  }
  if (num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("num_heads must be divisible by num_kv_heads for GQA.");
  }
  if ((hidden_size / num_heads) % 2 != 0) {
    throw std::invalid_argument("head_dim must be even for RoPE pairs.");
  }
}
} // namespace gpulab::transformer
