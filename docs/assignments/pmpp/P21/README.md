# P21：Dynamic Parallelism：设备启动子 Kernel（选做）

## 你要学会什么

dynamic parallelism 是 GPU kernel 在执行中启动子 kernel 的机制。通过一个小任务比较设备发起和 CPU 发起的代价，不把它当作默认更快的方案。

## 开始前

前置：P15、P20。阅读 PMPP 中关于CUDA dynamic parallelism、设备运行时和独立编译的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

本题选做。环境不支持时按本题要求交设计和限制说明，不阻塞主线。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

子任务对互不重叠的 float 区间做固定逐元素变换。父 kernel 只由指定线程每任务启动一次子 kernel，结果在主机等待父子整个任务完成后检查。不依赖父线程与子线程的同时执行，不使用旧教程中已移除的设备同步写法。

### 创建或修改的文件

- benchmarks/p21/dynamic.h：接口与合同。
- benchmarks/p21/dynamic_cpu.cpp：CPU reference。
- benchmarks/p21/dynamic.cu：GPU baseline 与本题优化版。
- benchmarks/p21/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p21.md 和 reports/results/p21/：报告与原始日志。

## 按步骤完成

1. 先提交父子任务图和 CPU reference，说明每个启动者及子任务写入区域。
2. 建立 CPU 连续启动子任务的 CUDA 基线，验证任务数 1/8/64、每任务元素 1/256/4096。
3. 单独建立本题 CUDA target，按安装的 Toolkit 文档启用设备代码链接并链接设备运行时；先完成一个父 kernel 启动一个子 kernel 的最小实验。
4. 扩展到相同任务集的父子版；主机最终同步、回读并逐元素验证，记录启动失败。不能把整个项目的编译设置为本题要求。
5. 预热 20 次、测量 100 次，比较完整父子执行与主机 launch 基线；硬件、权限或 Toolkit 不支持时只提交设计、伪代码、错误日志和替代方案，不阻塞后续题。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p21_test 和 p21_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p21_test p21_bench --parallel
mkdir -p reports/results/p21
set -o pipefail
./build/cuda-sm89/bin/p21_test | tee reports/results/p21/tests.txt
./build/cuda-sm89/bin/p21_bench | tee reports/results/p21/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

记录总 ms、任务数、每任务大小、launch 结构、代码链接选项和相对基线额外成本。不要把含父开销的耗时当子 kernel 纯计算时间。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 选做：用 nsys 核对工具版本实际支持的父子 launch 可见性，区分父任务总延迟和可见子任务，缺失轨道写未采集。
2. 保存设备链接/ptxas 资源日志与错误日志；nsys/ncu 对 dynamic parallelism 的支持随版本变化，不把未捕获子 kernel 当作未执行。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 支持时结果与 CPU 一致且每任务仅一次启动。
- 完整计时包含父和所有子任务，资源限制与错误被报告。
- 不支持时有明确证据和主机队列替代设计，标为未测。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 父 block 所有线程都 launch，意外产生大量重复任务。
- 父线程等待子线程使用不受支持的设备同步。
- 没有启用设备链接导致链接错误。

## API 参考

[CUDA Dynamic Parallelism 官方说明](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html)

API 和限制以实际安装的 Toolkit 版本为准；阅读对应版本文档后再接入，不自动升级环境。
