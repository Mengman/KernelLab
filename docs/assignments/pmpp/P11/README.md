# P11：Exclusive scan 与稳定压缩

## 你要学会什么

exclusive scan 是输出当前位置之前元素的和，例如 [2,3,1] 得到 [0,2,5]。用它为满足条件的元素生成互不冲突的写入位置，构造稳定压缩。

## 开始前

前置：P10。阅读 PMPP 中关于前缀和、工作高效 scan、多阶段分解的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

scan 输入 uint32，保证总和不溢出；空输入为空。压缩 float 输入中严格大于 0 的元素，保留原先顺序，返回元素数量；先提供容量 n 的输出，不计分配。n=0 输出长度为 0。

### 创建或修改的文件

- benchmarks/p11/scan_compact.h：接口与合同。
- benchmarks/p11/scan_compact_cpu.cpp：CPU reference。
- benchmarks/p11/scan_compact.cu：GPU baseline 与本题优化版。
- benchmarks/p11/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p11.md 和 reports/results/p11/：报告与原始日志。

## 按步骤完成

1. 写 CPU exclusive scan 与压缩，手算 flags=[0,1,1,0,1] 的位置和输出数量。
2. 实现单 block scan，先支持小输入，说明读取、上扫/下扫或其他算法中的同步点。
3. 实现多 block：块内 scan、块总和 scan、加块偏移；先允许块总和递归或多次 launch，不要求一步完成。
4. 压缩分成判定 flags、scan、scatter 写入三个阶段；输出数量要结合最后位置和最后 flag，空输入不能读最后元素。
5. 测试 n=0、1、31、33、257、1000、100003，全保留/全丢弃/交替/随机；测 n=2^20、2^24，保留比例 0%、50%、100%。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p11_test 和 p11_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p11_test p11_bench --parallel
mkdir -p reports/results/p11
set -o pipefail
./build/cuda-sm89/bin/p11_test | tee reports/results/p11/tests.txt
./build/cuda-sm89/bin/p11_bench | tee reports/results/p11/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

scan 与完整压缩分别报告 ms、输入元素/s；完整压缩含 flags、scan、scatter 所有阶段。验证整数位置精确一致，压缩数组顺序也一致。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 对 n=257、100003 等跨块样例跑 memcheck/initcheck，块内 scan 跑 racecheck/synccheck，验证临时偏移在读取前写入。
2. 用 NVTX/nsys 区分 flags、块 scan、块总和 scan、偏移和 scatter，找出全流程最耗时阶段，再只对该 kernel 做 ncu。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- exclusive 而非 inclusive，空输入安全。
- 跨 block 偏移正确，输出数量和稳定顺序与 CPU 一致。
- 性能包含完整算法，不只测最终 scatter。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 忘加前面 block 的总和：在块边界测试。
- 全丢弃输出长度计算读越界。
- 原子追加得到非稳定顺序：那不满足本题合同。
