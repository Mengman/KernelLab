add_library(gpulab_test_support STATIC random.cpp)
add_library(gpulab::test_support ALIAS gpulab_test_support)
target_include_directories(gpulab_test_support PUBLIC "${PROJECT_SOURCE_DIR}/include")
kernellab_target_options(gpulab_test_support)
