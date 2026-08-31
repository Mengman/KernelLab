# KernelLab

GPU Kernel Lab + Tiny Transformer inference project. Development targets Linux with CMake and C++20/CUDA C++20.
It supports separate native builds for Linux x86_64 (RTX 4090D) and Linux aarch64 (GB10); B200-specific
work is an independent elective.

This is a buildable, testable scaffold rather than a completed operator library or inference engine.
CUDA algorithms, performance experiments, and model execution are intentionally left to the chapter
assignments.

## Current capabilities

- Target-based CMake with independent CPU and CUDA configurations; no automatic downloads.
- `gpulab::runtime`: version information, device inspection, and a CPU-only fallback.
- `gpulab::kernels`: explicit integration target for completed assignments; no kernels registered yet.
- `gpulab::transformer`: model configuration and validation; no forward/decode implementation yet.
- `device_inspect`: text/JSON device summary with a strict GPU-required mode.
- `gpulab_bench --list`: benchmark entry point that currently reports no workloads.
- `tiny_transformer --print-config`: default learning-model configuration.
- CTest unit tests, CLI contract tests, optional GPU smoke tests, and Linux CPU CI.

## 1. Linux development environment

CPU builds require CMake >= 3.24, GNU Make, and a C++20 compiler. Python 3 enables optional JSON CLI
contract tests. On Ubuntu 24.04, install the base tools with:

```bash
sudo apt update
sudo apt install build-essential cmake python3
```

CUDA builds additionally require an NVIDIA driver and the Linux CUDA Toolkit:

- CUDA C++20 requires Toolkit 12.0 or newer.
- The GB10 preset requires a Toolkit that can compile `sm_121` (verify with `nvcc --list-gpu-code`).
- Use a host compiler supported by the selected Toolkit; do not hide compatibility issues with
  `--allow-unsupported-compiler`.
- `nvidia-smi` reports driver capability; it does not imply that `nvcc` is installed.
- WSL2 is supported for development. Build inside the Ubuntu shell with Linux tools; do not install
  a Linux display driver inside WSL.
- If the GPUs are on different hosts, build on each host and do not reuse build trees or ELF files.

Inspect the environment with:

```bash
uname -m
cmake --version
c++ --version
nvcc --version
nvcc --list-gpu-code
nvidia-smi
```

The last three commands are not needed for CPU-only builds. Presets use Unix Makefiles, so Ninja is
not required.

## 2. CPU-only scaffold validation

From the repository root:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug

./build/cpu-debug/bin/device_inspect --json
./build/cpu-debug/bin/gpulab_bench --list
./build/cpu-debug/bin/tiny_transformer --print-config
```

CPU-only device output should contain `cuda_compiled: false`, `status: disabled`, and an empty
`devices` array. This validates infrastructure only; it does not simulate GPU correctness or speed.

## 3. RTX 4090D and GB10

On the RTX 4090D host:

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --parallel
ctest --preset cuda-sm89
./build/cuda-sm89/bin/device_inspect --json --require-gpu
```

On the GB10 host:

```bash
cmake --preset cuda-sm121
cmake --build --preset cuda-sm121 --parallel
ctest --preset cuda-sm121
./build/cuda-sm121/bin/device_inspect --json --require-gpu
```

Override a compiler path before the first configure if necessary:

```bash
cmake --preset cuda-sm121 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

Use a new build directory or `cmake --fresh --preset <preset>` after changing a compiler or Toolkit.
Do not reuse a cache across toolchains.

## 4. Presets and switches

| Preset | Configuration | Target |
|---|---|---|
| `cpu-debug` | Debug, CUDA OFF | Daily unit tests |
| `cpu-release` | Release, CUDA OFF | Release regression |
| `cuda-debug` | RelWithDebInfo, native CUDA architecture | Host debugging |
| `cuda-release` | Release, native CUDA architecture | Current GPU |
| `cuda-sm89` | Release, CUDA 89 | RTX 4090D |
| `cuda-sm121` | Release, CUDA 121 | GB10 |

Every configure preset has matching build and test presets. Output goes to `build/<preset>/bin/`.
`native` requires a GPU during configuration; use explicit `89` or `121` on a GPU-less compile host
that still has the Toolkit.

Important cache options are `KERNELLAB_ENABLE_CUDA`, `BUILD_TESTING`,
`KERNELLAB_ENABLE_GPU_TESTS`, `KERNELLAB_BUILD_TOOLS`, `KERNELLAB_BUILD_BENCHMARKS`,
`KERNELLAB_BUILD_TRANSFORMER`, and `CMAKE_CUDA_ARCHITECTURES`. CUDA was deliberately requested
explicitly: a missing `nvcc` fails with a diagnostic instead of silently falling back to CPU.

## 5. Directory and integration rules

```text
cmake/                         CMake helpers and generated headers
include/gpulab/                Public APIs without mandatory CUDA headers
src/runtime/                   Host runtime and optional CUDA diagnostics
src/kernels/common/            Portable implementations and CPU references
src/kernels/sm89/              RTX 4090D optimizations
src/kernels/sm121/             GB10 optimizations
src/kernels/sm100a_optional/   B200 elective reservation, not built today
tools/device_inspect/          Device inventory executable
benchmarks/configs/             Workload configuration reservation
transformer/                   Tiny Transformer configuration and CLI
tests/                         Unit, CLI, and GPU smoke tests
reports/                       Versioned reports; raw results are ignored
```

Do not use source globbing for incomplete assignments. Add each completed kernel explicitly to a
CMake target, keep CUDA-only headers inside the CUDA backend, and keep B200 targets out of the main
fat binary until B01.

## 6. Tests and exit codes

GPU smoke validates allocation, one scalar launch, synchronization, and a copy back. It is not a
throughput benchmark. With no usable GPU it returns 77 and CTest marks it skipped; this is not a
successful GPU validation. Other CUDA errors fail the test.

`device_inspect` returns 0 for a completed inspection, 1 for an argument/runtime error, and 2 when
`--require-gpu` cannot be satisfied. JSON status values are `disabled`, `unavailable`, and `ready`.

`tiny_transformer --generate` and `gpulab_bench --run gemm` currently fail explicitly because those
implementations are future assignments; they never emit fabricated measurements.


## Official references

- [CMake CUDA architectures](https://cmake.org/cmake/help/v3.30/prop_tgt/CUDA_ARCHITECTURES.html)
- [CUDA C++20 support](https://developer.nvidia.com/blog/cuda-toolkit-12-0-released-for-general-availability/)
- [CUDA Linux installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
