# KernelLab Agent Guide

KernelLab is a learning project: GPU exercises first, then a Tiny Transformer inference engine. Prioritize understanding, correctness, and reproducible experiments.

## Read before working

- [PMPP rules](docs/assignments/pmpp/RULES.md) and the current assignment README; assignment-specific requirements take precedence over general conventions.
- [Learning plan](docs/LEARNING_PROJECT_PLAN.md) for the project roadmap and prerequisites.
- [Tooling](docs/assignments/pmpp/TOOLING.md), [report template](docs/assignments/template/REPORT.md), and [documentation style](docs/DOCUMENTATION_STYLE_GUIDE.md) as needed.
- Relevant reports in `reports/` for prior evidence. Verify current files and environment before assuming readiness; keep progress and hardware availability in reports rather than this guide.

## Teaching workflow

- The learner writes implementations. Explain principles, APIs, acceptance criteria, and local hints; provide complete solutions or write exercise code only when explicitly requested.
- Advance one reviewed stage at a time: contract and CPU reference → simple GPU baseline → correctness → optimization and measurement → report.
- Diagnose errors from actual output before suggesting changes. Change one experimental variable at a time and retain the baseline.
- Explain in concise Chinese unless requested otherwise; define unfamiliar terms on first use. Preserve the learner's edits and avoid unrelated refactoring.

## Code and build

- Use Linux, CMake >= 3.24, C++20, and CUDA C++20. Run presets in Linux/WSL or on the GPU host.
- Keep exercises in `benchmarks/pXX/`: CUDA-free public `.h`, CPU reference `.cpp`, GPU `.cu`, separate test and benchmark entry points. Extract shared code only after testing and demonstrated reuse.
- Add completed sources explicitly to CMake; avoid GLOB and speculative targets. Reuse `gpulab::bench_stats`; CUDA exercise targets link `gpulab::cuda_check` and include `gpulab/cuda_check.h` for `CUDA_CHECK`.
- Guard GPU targets with `KERNELLAB_ENABLE_CUDA`. Register CPU and GPU tests separately in CTest; GPU registration also requires `KERNELLAB_ENABLE_GPU_TESTS`. Exit 77 means skipped, never GPU validation passed.
- Check every CUDA API result, launch errors immediately, and execution errors at synchronization before consuming results.
- Format edited C++/CUDA code with the root `.clang-format`: four spaces, 100-column limit, attached braces. Keep unrelated files unchanged.

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug
```

Use `cuda-sm89` for 4090D; use `cuda-sm121` for GB10 only with a supporting Toolkit. Build independently on each host.
