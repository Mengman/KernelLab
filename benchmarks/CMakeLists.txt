add_executable(gpulab_bench main.cpp)
target_link_libraries(gpulab_bench PRIVATE gpulab::kernels)
kernellab_target_options(gpulab_bench)

if(KERNELLAB_ENABLE_CUDA)
	add_library(p01_fma STATIC p01/fma.cu)
	target_link_libraries(p01_fma PUBLIC CUDA::cudart)
	kernellab_target_options(p01_fma)

	add_executable(p01_test_fma p01/test_fma.cpp)
	target_link_libraries(p01_test_fma PRIVATE p01_fma)
	kernellab_target_options(p01_test_fma)

	add_executable(p01_bench_h2d p01/bench_h2d.cpp)
	target_link_libraries(p01_bench_h2d PRIVATE CUDA::cudart)
	target_link_libraries(p01_bench_h2d PRIVATE gpulab::bench_stats)
	kernellab_target_options(p01_bench_h2d)
endif()