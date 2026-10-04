# P22：CUDA Graph 与 Workspace 复用

## 你要学会什么

CUDA Graph 是可反复提交的 GPU 工作依赖图；workspace 是提前分配并重复使用的临时存储。分别观察减少重复 launch 与避免重复显存分配的效果，为固定推理流程准备执行方式。

## 开始前

前置：P16、P19、P20。阅读 PMPP 中关于CUDA Graph、stream capture、内存生命周期的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

复用 P16 两层 MLP，固定 shape、权重和显存地址。先在 GPU 上完成整个前向，不在捕获区内分配、释放或做主机同步。Graph 输出必须随输入变化；不是缓存旧结果。shape 改变先销毁旧图并重建，不做图更新优化。

### 创建或修改的文件

- benchmarks/p22/graph_workspace.h：接口与合同。
- benchmarks/p22/graph_workspace_cpu.cpp：CPU reference。
- benchmarks/p22/graph_workspace.cu：GPU baseline 与本题优化版。
- benchmarks/p22/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p22.md 和 reports/results/p22/：报告与原始日志。

## 按步骤完成

1. 实现三版：A 每次分配+普通 launch，B 复用 workspace+普通 launch，C 同一 workspace+Graph replay；数学工作、GEMM 和融合选项相同。
2. 对固定 shape 创建非阻塞 stream，先完成普通执行和正确性，再捕获固定 kernel 序列、实例化 Graph；捕获时不做 cudaDeviceSynchronize/cudaEventSynchronize，错误检查不能隐式触发同步。
3. 用两组不同输入交替 replay，分别与 CPU 比较；记录 workspace 生命周期和每个指针归属。
4. 测试 shape (1,128,256,64)、(32,256,512,128)、(127,129,257,65)，切换 shape 时重建，验证空 batch 路径。
5. 将捕获/实例化开销单列；固定 shape 稳态预热 20 次、测量 100 次。CPU 墙钟测 A/B/C 全执行，B/C 另测 GPU 时间，报告多少次 replay 后可能摊销建图成本。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p22_test 和 p22_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p22_test p22_bench --parallel
mkdir -p reports/results/p22
set -o pipefail
./build/cuda-sm89/bin/p22_test | tee reports/results/p22/tests.txt
./build/cuda-sm89/bin/p22_bench | tee reports/results/p22/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告 CPU 端到端 ms、GPU ms、一次性建图/实例化时间、workspace bytes 和 launch 数。A/B 的分配开销用 CPU 墙钟观察，不能声称 Event 精确测到了主机分配时间。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. NVTX/nsys 分开分配、捕获、实例化、首次 replay 与稳态 replay；按版本选择 graph/node 粒度并记录额外开销。
2. 只用普通 kernel 路径上的 ncu 分析具体计算瓶颈，Graph 性能改进主要用正常端到端测量与 nsys 比较；检查 capture 内没有隐藏同步。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 不同输入 replay 正确，shape 改变不复用失效图。
- A/B/C 工作相同，分配收益与 Graph 收益分别比较。
- 报告捕获支持条件、构建成本和稳态结果，不省略首次成本。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 捕获内部使用会同步的包装函数。
- 捕获后显存指针已释放或换地址。
- 只测 replay 却宣称首次执行也更快。

## API 参考

[CUDA Graph 捕获规则](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

API 和限制以实际安装的 Toolkit 版本为准；阅读对应版本文档后再接入，不自动升级环境。
