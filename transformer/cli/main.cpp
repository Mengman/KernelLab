#include "gpulab/transformer/config.h"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--print-config") {
    try {
      const gpulab::transformer::ModelConfig config;
      config.validate();
      std::cout << "{\"vocab_size\":" << config.vocab_size << ",\"hidden_size\":" << config.hidden_size
                << ",\"num_layers\":" << config.num_layers << ",\"num_heads\":" << config.num_heads
                << ",\"num_kv_heads\":" << config.num_kv_heads
                << ",\"head_dim\":" << config.hidden_size / config.num_heads
                << ",\"intermediate_size\":" << config.intermediate_size
                << ",\"max_seq_len\":" << config.max_seq_len << "}\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--help")) {
    std::cout << "Usage: tiny_transformer --print-config\n"
                 "Configuration skeleton only. Inference is not implemented.\n";
    return 0;
  }
  std::cerr << "Unsupported command. Inference is not implemented yet.\n";
  return 1;
}
