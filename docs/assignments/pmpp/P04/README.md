# P04：分支、线程块大小与占用率

## 你要学会什么

用一次只改变一个参数的实验，区分线程块大小、分支模式和输入分布的影响。占用率是 SM 上驻留线程数相对硬件上限的比例，高占用率并不保证更快。

## 开始前

前置：P02、P03。阅读 PMPP 中关于warp 执行、分支发散、寄存器和占用率的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float 向量逐元素处理：x>=0 时重复 64 次 v=0.99*v+0.01，否则重复 64 次 v=0.98*v-0.01；out 写最终 v。先固定循环次数，不使用未定义的输入。分支发散指同一 warp 内线程走不同路径；CPU 与 GPU 使用同一公式。

### 创建或修改的文件

- benchmarks/p04/branch.h：接口与合同。
- benchmarks/p04/branch_cpu.cpp：CPU reference。
- benchmarks/p04/branch.cu：GPU baseline 与本题优化版。
- benchmarks/p04/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p04.md 和 reports/results/p04/：报告与原始日志。

## 按步骤完成

1. 写 CPU reference 与保留实际分支的 GPU baseline，验证 0、1、31、32、33、1000；编译器可能合并或改写分支，不能只凭源码推断发散。
2. 生成全正、前半正后半负、正负交替、随机正负四类输入，保持长度和数值范围相同。
3. 固定输入与算法，分别测 64、128、256、512 threads/block。grid 用同一覆盖规则，记录实际 block 数。
4. 对 n=2^20、2^24 测四类分布；输出最小值、中位数及 ns/元素，不交叉改变循环次数。
5. 用编译器资源报告记录每线程寄存器与每 block shared memory；能使用 Nsight Compute（kernel 性能分析工具）时记录占用率和分支相关证据，不可用则写当前环境不支持，采用资源上限分析。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p04_test 和 p04_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p04_test p04_bench --parallel
mkdir -p reports/results/p04
set -o pipefail
./build/cuda-sm89/bin/p04_test | tee reports/results/p04/tests.txt
./build/cuda-sm89/bin/p04_bench | tee reports/results/p04/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

主要报告 ms、元素/s；不将路径不同的复杂循环随意换算成统一 GFLOP/s。资源与 profiler 数据的版本、指标名和命令一起记录。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 对同一输入的 64/128/256/512 threads/block 保存 ptxas 资源日志，分别采集 ncu launch/occupancy 信息；比较正常运行时间与资源限制。
2. 选择全正与正负交替两个配置，结合源码对应的 SASS/谓词/分支和实际活跃线程信息分析；不能由源码 if 直接认定一定发散。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 所有 block 大小和分布都通过正确性。
- 比较四种分布时只改变分支分布，比较 block 时只改变 block 大小。
- 结论区分源码预期、编译后证据和实际时间；允许没有明显分支差异。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 将两个分支写成完全相同工作：编译器可能消除分支。
- 计时中初始化不同输入：会混入主机工作。
- 只根据占用率排序性能：还需时间和资源证据。
