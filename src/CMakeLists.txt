add_library(gpulab_runtime STATIC runtime/build_info.cpp)
add_library(gpulab::runtime ALIAS gpulab_runtime)
target_include_directories(gpulab_runtime PUBLIC
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_BINARY_DIR}/generated"
)
kernellab_target_options(gpulab_runtime)
if(KERNELLAB_ENABLE_CUDA)
  target_sources(gpulab_runtime PRIVATE runtime/cuda/device.cpp runtime/cuda/smoke.cu)
  target_link_libraries(gpulab_runtime PRIVATE CUDA::cudart)
  set_target_properties(gpulab_runtime PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
  target_compile_options(gpulab_runtime PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>")
else()
  target_sources(gpulab_runtime PRIVATE runtime/device_stub.cpp)
endif()
add_subdirectory(kernels)
