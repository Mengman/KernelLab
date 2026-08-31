add_executable(device_inspect device_inspect/main.cpp)
target_link_libraries(device_inspect PRIVATE gpulab::runtime)
kernellab_target_options(device_inspect)
