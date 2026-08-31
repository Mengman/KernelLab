add_executable(gpulab_bench main.cpp)
target_link_libraries(gpulab_bench PRIVATE gpulab::kernels)
kernellab_target_options(gpulab_bench)
