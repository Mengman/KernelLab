# P20：多 Stream 分块流水线

## 你要学会什么

stream 是按顺序执行任务的队列，不同 stream 可能并发。将 H2D（CPU 到 GPU复制）、计算、D2H（GPU 到CPU复制）按块流水执行，比较单流与多流的端到端时间。

## 开始前

前置：P02、P19。阅读 PMPP 中关于CUDA streams、固定页内存、事件依赖和并发复制的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

复用 P02 的逐元素运算，float 输入，chunk 即一段连续元素；最后一段允许不足整块。每个同时在途任务拥有独立 CPU 固定页缓冲区和显存工作区，复用前等待完成，输出保持原顺序。

### 创建或修改的文件

- benchmarks/p20/pipeline.h：接口与合同。
- benchmarks/p20/pipeline_cpu.cpp：CPU reference。
- benchmarks/p20/pipeline.cu：GPU baseline 与本题优化版。
- benchmarks/p20/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p20.md 和 reports/results/p20/：报告与原始日志。

## 按步骤完成

1. 查询并记录设备并发能力和 asyncEngineCount（异步复制引擎数量）；它表示能力，不能证明实际重叠。
2. 先实现单 stream 分块的完整 H2D→kernel→D2H，验证 n=0、1、1000、chunk+1，计时包含整段传输与计算。
3. 创建 2/4 个非阻塞 stream，分别管理事件和缓冲区；复用槽位前等待该槽位完成，不在每个 chunk 后调用全设备同步。
4. 固定总输入每数组 64 MiB，比较 chunk=256 KiB、1 MiB、8 MiB 与 stream 数 1/2/4，每次只改一个参数。输出与 CPU 逐元素比较。
5. 用 Nsight Systems（CPU/GPU 时间线工具）观察 H2D、kernel、D2H 是否重叠；不可用则写未确认实际重叠。输出 host wall time 与 GPU 整段 Event 时间，协调开始和结束的跨 stream 依赖。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p20_test 和 p20_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p20_test p20_bench --parallel
mkdir -p reports/results/p20
set -o pipefail
./build/cuda-sm89/bin/p20_test | tee reports/results/p20/tests.txt
./build/cuda-sm89/bin/p20_bench | tee reports/results/p20/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告端到端中位 ms、元素/s，并单列一次性分配时间。CPU 使用单调时钟包围提交到最终全部完成；GPU Event 必须等待所有工作流终点，不能只包一个流。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. nsys 时间线是本题核心证据：标出 chunk/stream，确认 H2D、kernel、D2H 实际重叠及槽位复用等待，统计端到端范围。
2. ncu 常规单 kernel 采集可能串行化执行，不用它证明多流重叠或评价原流水线并发；Sanitizer 和数值测试检查尾 chunk，依赖正确性仍需事件协议说明。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 尾 chunk 和缓冲区复用正确，无提前覆盖源或读取未完成输出。
- 端到端计时覆盖全部流和传输，单流与多流输入相同。
- 重叠结论有时间线或明确写未确认，速度不提升也如实解释。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- CPU 内存未固定页导致复制行为不符合预期。
- 默认 stream 的隐式依赖破坏并发。
- 只等待一个 stream 就停止计时。

## API 参考

[CUDA 并发复制说明](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation)

API 和限制以实际安装的 Toolkit 版本为准；阅读对应版本文档后再接入，不自动升级环境。
