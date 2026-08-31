# Explicit preset values take priority over environment and local GPU detection.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" CACHE STRING "CUDA architectures")
  else()
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
  endif()
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "CUDA was requested but nvcc was not found. Install a supported Linux CUDA toolkit, "
    "set -DCMAKE_CUDA_COMPILER=/path/to/nvcc, or use cmake --preset cpu-debug. "
    "An installed NVIDIA driver alone is not a CUDA toolkit.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit 12.0 REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
message(STATUS "CUDA toolkit ${CUDAToolkit_VERSION}; architectures=${CMAKE_CUDA_ARCHITECTURES}")
