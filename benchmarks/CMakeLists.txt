add_executable(gpulab_bench main.cpp)
target_link_libraries(gpulab_bench PRIVATE gpulab::kernels)
kernellab_target_options(gpulab_bench)

if(KERNELLAB_ENABLE_CUDA)
    set_source_files_properties(p01/main.cpp PROPERTIES LANGUAGE CUDA)

    add_executable(p01_bench p01/main.cpp)
    target_link_libraries(p01_bench PRIVATE gpulab::kernels)
    kernellab_target_options(p01_bench)
endif()