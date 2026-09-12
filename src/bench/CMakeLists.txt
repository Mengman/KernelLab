add_library(gpulab_bench_stats STATIC stats.cpp)
add_library(gpulab::bench_stats ALIAS gpulab_bench_stats)
target_include_directories(gpulab_bench_stats PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
)
kernellab_target_options(gpulab_bench_stats)