#include "gpulab/build_info.h"

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view{argv[1]} == "--list") {
    std::cout << "KernelLab " << gpulab::version() << "\n"
                 "No benchmark cases registered yet. Start with assignment P01.\n";
    return 0;
  }
  if (argc == 1 || (argc == 2 && std::string_view{argv[1]} == "--help")) {
    std::cout << "Usage: gpulab_bench --list\n"
                 "Scaffold only: timing and workload registration are future assignments.\n";
    return 0;
  }
  std::cerr << "Unsupported benchmark arguments. No workloads have been implemented.\n";
  return 1;
}
