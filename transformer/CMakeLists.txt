add_library(tiny_transformer_model STATIC model/config.cpp)
add_library(gpulab::transformer ALIAS tiny_transformer_model)
target_include_directories(tiny_transformer_model PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(tiny_transformer_model PUBLIC gpulab::kernels)
kernellab_target_options(tiny_transformer_model)

add_executable(tiny_transformer cli/main.cpp)
target_link_libraries(tiny_transformer PRIVATE gpulab::transformer)
kernellab_target_options(tiny_transformer)
