# KernelLab：项目说明（白话版）

这是一个边读书边写代码的学习项目：先练 GPU 小算子，再把它们组合成 Tiny Transformer 推理引擎。
开发环境固定为 Linux + CMake + C++20 / CUDA C++20；4090D 和 GB10 分别在自己的 Linux 主机上构建。
B200 专属内容是选作题，不会阻塞主线。

如果你第一次打开项目，建议按这个顺序：

1. 阅读 [学习计划](LEARNING_PROJECT_PLAN.md) 第 1 节，先看术语解释。
2. 阅读 [P01 作业](assignments/pmpp/P01/README.md)，按步骤完成第一个实验。
3. 用 [报告模板](assignments/template/REPORT.md) 记录环境、正确性和性能。
4. 后续由 AI 或开发者新增文档时，遵守 [文档写作规范](DOCUMENTATION_STYLE_GUIDE.md)。

文档中的 H2D/D2H、FMA throughput、occupancy 等词都会在第一次出现时解释。“性能账本”只是记录实验数字的表格，不是额外工具。

目前是可构建、可测试的工程框架，不是完成的算子库或推理引擎。CUDA 算法、性能计时和模型执行留给章节作业。

## 已有能力

- target-based CMake 与独立的 CPU/CUDA 配置，构建时不下载第三方依赖。
- `gpulab::runtime`：版本信息、设备探测接口、CPU-only fallback。
- `gpulab::kernels`：章节算子的接入 target，目前未注册算法。
- `gpulab::transformer`：模型配置与校验，目前没有 forward/decode 实现。
- `device_inspect`：文本/JSON 设备摘要，支持严格要求 GPU。
- `gpulab_bench --list`：benchmark 入口，目前明确返回空 workload 列表。
- `tiny_transformer --print-config`：输出学习模型默认配置。
- CTest 单元测试、CLI 契约测试、可选 GPU 冒烟测试和 Linux CPU CI 配置。

## 1. Linux 开发环境

CPU 构建需要：CMake >= 3.24、GNU Make、支持 C++20 的 C++ 编译器；Python 3 用于可选 CLI JSON 测试。
Ubuntu 24.04 可安装以下基础工具（由开发者自行执行）：

```bash
sudo apt update
sudo apt install build-essential cmake python3
```

CUDA 构建还需要 NVIDIA 驱动和 **Linux CUDA Toolkit**：

- 项目使用 CUDA C++20，因此最低 Toolkit 为 12.0。
- GB10 preset 需要能够编译 `sm_121` 的 Toolkit（12.9 或更新的支持版本）；用 `nvcc --list-gpu-code` 验证。
- host compiler 必须与所用 Toolkit 匹配，不建议通过 `--allow-unsupported-compiler` 掩盖环境问题。
- `nvidia-smi` 显示驱动兼容的 CUDA 版本，不代表系统安装了 `nvcc`。
- WSL2 可用于开发；在 Ubuntu shell 内运行构建，不使用 Windows 版 CMake/nvcc。不要在 WSL 内安装 Linux 显示驱动。
- GB10 与 4090D 若在不同主机，各自执行构建；不跨机器复用 `build/` 或 ELF 可执行文件。

验证环境：

```bash
uname -m
cmake --version
c++ --version
nvcc --version
nvcc --list-gpu-code
nvidia-smi
```

CPU-only 构建不需要最后三项。CMake preset 使用 Unix Makefiles，无需 Ninja。

## 2. 无 CUDA 的 CPU 框架验证

在仓库根目录：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --parallel
ctest --preset cpu-debug

./build/cpu-debug/bin/device_inspect --json
./build/cpu-debug/bin/gpulab_bench --list
./build/cpu-debug/bin/tiny_transformer --print-config
```

CPU-only 的设备输出应为 `cuda_compiled: false`、`status: disabled`、`devices: []`。
这用于验证工程基础设施，不是模拟 GPU，也不能用它验证 CUDA 正确性或性能。

## 3. 4090D 和 GB10

4090D Linux 主机：

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --parallel
ctest --preset cuda-sm89
./build/cuda-sm89/bin/device_inspect --json --require-gpu
```

GB10 Linux 主机：

```bash
cmake --preset cuda-sm121
cmake --build --preset cuda-sm121 --parallel
ctest --preset cuda-sm121
./build/cuda-sm121/bin/device_inspect --json --require-gpu
```

需要指定 Toolkit/host compiler 时，例如：

```bash
cmake --preset cuda-sm121 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

如需指定 `CMAKE_CUDA_HOST_COMPILER`，在第一次配置前传入，并选取受该 Toolkit 支持的编译器。
更换编译器/Toolkit 使用新 build 目录或 `cmake --fresh --preset <preset>`，不要跨工具链复用旧 cache。

## 4. 构建预设与开关

| Preset | 配置 | 目标 |
|---|---|---|
| `cpu-debug` | Debug，CUDA OFF | 日常单元测试 |
| `cpu-release` | Release，CUDA OFF | 优化构建下的基础回归 |
| `cuda-debug` | RelWithDebInfo，CUDA native | host 调试符号与 CUDA 行号 |
| `cuda-release` | Release，CUDA native | 当前主机 GPU |
| `cuda-sm89` | Release，CUDA 89 | RTX 4090D |
| `cuda-sm121` | Release，CUDA 121 | GB10 |

每个 preset 都有同名 configure/build/test preset。输出统一在 `build/<preset>/bin/`。
`native` 要求配置时能探测本地 GPU；无 GPU 的编译主机使用显式 `89`/`121`，但仍需要 Toolkit。

| CMake cache 选项 | 默认 | 作用 |
|---|---|---|
| `KERNELLAB_ENABLE_CUDA` | ON | 启用 CUDA 编译；找不到 nvcc 明确失败，不偷偷切换 CPU |
| `BUILD_TESTING` | ON | CTest 测试 |
| `KERNELLAB_ENABLE_GPU_TESTS` | OFF | 注册 GPU smoke；CUDA presets 设置为 ON |
| `KERNELLAB_BUILD_TOOLS` | ON | 设备探测工具 |
| `KERNELLAB_BUILD_BENCHMARKS` | ON | benchmark 入口 |
| `KERNELLAB_BUILD_TRANSFORMER` | ON | Transformer 配置库与 CLI |
| `CMAKE_CUDA_ARCHITECTURES` | native | 显式覆盖 GPU 架构 |

例如，只构建 runtime 和 CPU tests：

```bash
cmake -S . -B build/minimal -DKERNELLAB_ENABLE_CUDA=OFF \
  -DKERNELLAB_BUILD_TOOLS=OFF -DKERNELLAB_BUILD_BENCHMARKS=OFF \
  -DKERNELLAB_BUILD_TRANSFORMER=OFF
cmake --build build/minimal --parallel
ctest --test-dir build/minimal --output-on-failure
```

机器私有路径可以放 `CMakeUserPresets.json`（已忽略）。不在仓库里硬编码 `/usr/local/cuda-*` 或用户 home。
初期不提供 install/package/ABI 稳定性承诺，后续章节按需增加。

## 5. 目录与接入规则

```text
cmake/                         构建辅助与生成头文件模板
include/gpulab/                不强制依赖 CUDA headers 的公共 API
src/runtime/                   host runtime 与可选 CUDA 诊断
src/kernels/common/            通用实现与 CPU reference（待作业）
src/kernels/sm89/              4090D 优化（待作业）
src/kernels/sm121/             GB10 优化（待作业）
src/kernels/sm100a_optional/   B200 选修预留，不参与当前构建
tools/device_inspect/          设备档案入口
benchmarks/configs/            workload 配置预留
transformer/model/             模型配置；模型本体待 T0–T6
transformer/cli/               CLI
tests/unit/                    C++ 测试，无外部测试框架依赖
tests/gpu/                     CUDA 集成冒烟测试
docs/assignments/              PMPP / Modern GPU / B200 题目入口（本地，不提交）
reports/                       报告；results/ 放未跟踪的原始测量
```

新增 kernel 时不要先做复杂 registry。先完成一个显式 target，再逐题抽取共用接口。例如未来 P02：

```cmake
# src/kernels/CMakeLists.txt — 完成 common/axpby.cu 后才加入
if(KERNELLAB_ENABLE_CUDA)
  add_library(gpulab_axpby STATIC common/axpby.cu)
  target_link_libraries(gpulab_axpby PUBLIC gpulab::runtime)
  target_link_libraries(gpulab_kernels INTERFACE gpulab_axpby)
endif()
```

- 不使用文件 GLOB 自动收集尚未完成的作业；source 显式列入 CMake。
- 测试单独建 target，放到 CTest；不要把测试入口写进公共库。
- `gpulab::kernels` 消费者继承已完成算子的链接依赖。
- CUDA 专属头文件只在 CUDA backend 内使用，CPU-only 不包含 CUDA headers。
- B200 专属 target 和工具链要求等 B01 再增加，不把 `sm_100a` 当作 GB10 fallback。
- PyTorch/TVM/TIRx 等依赖等相应作业再加入，不在初始构建自动下载。

## 6. 测试、设备诊断与退出码

```bash
ctest --preset cpu-debug
ctest --preset cuda-sm89 -L gpu
compute-sanitizer --tool memcheck ./build/cuda-sm89/bin/cuda_smoke_tests
```

GPU smoke 仅验证分配、一次标量 kernel launch、同步和拷回，不是吞吐量 benchmark。
无可用 GPU/驱动时该测试以 77 返回，CTest 显示 Skipped；这**不算 GPU 验证通过**。
其他 CUDA 错误作为失败处理。CI 仅运行 CPU Debug/Release，GPU 验证在实际主机执行。

`device_inspect`：0 表示完成检查，1 表示参数或运行时错误，2 表示 `--require-gpu` 无法满足。
JSON `schema_version=1` 的状态有 `disabled`（未编译 CUDA）、`unavailable`（无 GPU/驱动）、`ready`。

`tiny_transformer --generate`、`gpulab_bench --run gemm` 当前会明确报错，不产生虚构结果。
测试不依赖 `assert`，因此 Release/NDEBUG 下仍会检查。

## 7. 从哪里开始

下一步是 [P01](docs/assignments/pmpp/P01/README.md)：扩展设备档案并自行实现吞吐 microbenchmark。
模板见 [REPORT.md](docs/assignments/template/REPORT.md)。
完整题目设计见本地 [学习计划](docs/LEARNING_PROJECT_PLAN.md)。
注意：现有 `.gitignore` 的 `docs/` 忽略规则保留不变，学习计划目前是本地文件；新 clone 后需要单独复制，
或在你决定将它纳入版本管理后调整该规则。

## 官方参考

- [CMake CUDA_ARCHITECTURES / native](https://cmake.org/cmake/help/v3.30/prop_tgt/CUDA_ARCHITECTURES.html)
- [CUDA C++20 支持](https://developer.nvidia.com/blog/cuda-toolkit-12-0-released-for-general-availability/)
- [CUDA Linux 安装与 host compiler 支持](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
