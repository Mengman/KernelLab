# sm_89

RTX 4090D-specific optimizations. Add them to the build only when CUDA is enabled; the CPU runtime
must not directly reference architecture-specific implementations.
