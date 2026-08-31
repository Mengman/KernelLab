# Curriculum kernels will be added explicitly here as each assignment is completed.
# No globbing, automatic downloads, or unimplemented kernel advertised as usable.
add_library(gpulab_kernels INTERFACE)
add_library(gpulab::kernels ALIAS gpulab_kernels)
target_link_libraries(gpulab_kernels INTERFACE gpulab::runtime)
