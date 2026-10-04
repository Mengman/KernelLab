# P13：整数 Radix sort 与 Top-k

## 你要学会什么

radix sort 是按整数的若干位逐轮分桶的排序方法。复用 histogram 和 scan 构建排序，再比较全量排序取前 k 与候选集合选择，理解不同目标的工作量。

## 开始前

前置：P09、P11、P12。阅读 PMPP 中关于基数排序、稳定分桶、scan 与选择的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

初版 uint32 升序稳定排序，附带原下标标签；不处理有符号键或浮点 NaN。Top-k 取最大的 k 个值，输出降序，同值按原下标升序；k=0 为空，k>n 拒绝。

### 创建或修改的文件

- benchmarks/p13/sort_topk.h：接口与合同。
- benchmarks/p13/sort_topk_cpu.cpp：CPU reference。
- benchmarks/p13/sort_topk.cu：GPU baseline 与本题优化版。
- benchmarks/p13/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p13.md 和 reports/results/p13/：报告与原始日志。

## 按步骤完成

1. 用标准库写 CPU 参考，验证排序与 Top-k 的标签和重复值规则。
2. 先做每轮 1 bit 的稳定 radix sort，拆成判定/计数、scan 与稳定写入；32 轮覆盖全部位，不能只测试低位。
3. 多 block 版本计算各桶全局偏移和块内稳定位置；正确后可选 4 bits/轮，说明计数/偏移资源变化。
4. 实现全量排序取 Top-k 基线，再实现局部候选加合并的选择版；最终排序并按合同处理相等值。
5. 测试 n=0、1、33、257、1000，k=0、1、n、n+1；键含 0、UINT32_MAX、全重复、随机。测 n=2^20、2^24，k=1、32、256，记录选择版优势是否随 k 改变。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p13_test 和 p13_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p13_test p13_bench --parallel
mkdir -p reports/results/p13
set -o pipefail
./build/cuda-sm89/bin/p13_test | tee reports/results/p13/tests.txt
./build/cuda-sm89/bin/p13_bench | tee reports/results/p13/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

全排序与完整 Top-k 各报告总 ms、输入元素/s、临时显存容量，所有 radix 轮次和合并阶段都计时。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 用 NVTX/nsys 标出每轮计数/scan/scatter 与 Top-k 候选合并，统计完整算法 launch 和阶段占比。
2. 选一个关键阶段用 ncu 比较 1 bit/4 bits 或候选量的资源与访存，另列所有轮次总时间；memcheck 检查高位和非整齐输入。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 高位键、稳定标签和 Top-k 同值规则均正确。
- 非法 k 被拒绝，空输入不访问最后元素。
- 比较相同任务的全排序与候选选择，不把排序完整输出与 Top-k 当作同一产出。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 每轮分桶不稳定：多位排序无法保证正确。
- 只测小整数遗漏高位错误。
- 局部候选漏了某块中的多个大值：证明每块保留量足够。
