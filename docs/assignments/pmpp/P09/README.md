# P09：原子直方图与局部合并

## 你要学会什么

直方图统计每个取值出现次数；原子操作保证并发更新同一地址不丢失。比较全局原子更新和每个 block 的局部统计，观察输入冲突分布如何改变瓶颈。

## 开始前

前置：P04、P05。阅读 PMPP 中关于histogram、原子操作和共享内存的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

输入 uint8，256 个桶；输出 uint32，实验 n 不超过 uint32 计数上限。输出在每次计算前清零，空输入得到 256 个零；CPU 精确计数，GPU 不用浮点误差比较。

### 创建或修改的文件

- benchmarks/p09/histogram.h：接口与合同。
- benchmarks/p09/histogram_cpu.cpp：CPU reference。
- benchmarks/p09/histogram.cu：GPU baseline 与本题优化版。
- benchmarks/p09/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p09.md 和 reports/results/p09/：报告与原始日志。

## 按步骤完成

1. 写 CPU reference，手算 [0,1,1,255]；检查计数总和等于 n。
2. 实现每元素更新全局桶的 baseline；每次执行前清零输出，初始化不计入 kernel-only 时间。
3. 实现 shared-memory 局部桶，先协作清零、统计，再合并到全局桶；明确两个同步点。
4. 测试空、单元素、全相同、均匀覆盖 0..255、随机输入、n=257 和 1000。
5. 测 n=2^20、2^24、2^26，用全相同、均匀、90% 为同值三种分布；报告局部统计与合并合计时间。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p09_test 和 p09_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p09_test p09_bench --parallel
mkdir -p reports/results/p09
set -o pipefail
./build/cuda-sm89/bin/p09_test | tee reports/results/p09/tests.txt
./build/cuda-sm89/bin/p09_bench | tee reports/results/p09/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告 ms、输入 GB/s=n*sizeof(uint8)/时间、元素/s。若把清零包含在端到端时间中，另列结果，不与 kernel-only 混用。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 用 ncu 对同长度的全相同与均匀输入分别采基线/局部桶版，选择实际可用的原子或访存 section，明确不能取得的指标。
2. 用 nsys 验证清零在何处、局部统计和合并是否有独立 kernel；所有输出每轮清零且计时口径保持一致。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 每桶计数与 CPU 相同且总和为 n。
- 计时迭代不会累积旧计数。
- 三类冲突输入都有基线与优化实测，不能只测均匀随机。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 没有清零导致每轮计数累加。
- shared 桶初始化或合并之前缺少同步。
- 局部桶解决所有冲突：同 block 内仍有原子争用。
