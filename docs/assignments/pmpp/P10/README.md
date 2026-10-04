# P10：归约与 RMSNorm

## 你要学会什么

归约是把很多元素合成一个结果，例如求和或最大值。在完成 sum、max、平方和后实现 RMSNorm（按均方根缩放的归一化），为 Transformer 的归一化算子准备实现。

## 开始前

前置：P05、P09。阅读 PMPP 中关于并行归约、warp 通信、数值稳定性的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

归约输入 float，CPU 有限数参考使用 double 累积。sum 和平方和的空结果为 0，max 的空结果为 -Inf；任意 NaN 输入传播 NaN。RMSNorm 输入 [rows,D]，输出 x*gamma/sqrt(mean(x^2)+eps)，gamma 长度 D，eps>0；rows=0 返回空，D=0 拒绝。RMSNorm 非有限行传播 NaN 到整行，作为本题显式约定。

### 创建或修改的文件

- benchmarks/p10/reduce_norm.h：接口与合同。
- benchmarks/p10/reduce_norm_cpu.cpp：CPU reference。
- benchmarks/p10/reduce_norm.cu：GPU baseline 与本题优化版。
- benchmarks/p10/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p10.md 和 reports/results/p10/：报告与原始日志。

## 按步骤完成

1. 先写三种 CPU 归约，定义 NaN、Inf、负数和空输入规则，再写 RMSNorm CPU reference。
2. GPU 从单 block 归约开始，覆盖非 2 的幂；大输入使用局部结果与第二阶段归约，不尝试普通 kernel 内跨 block barrier。
3. 验证 n=0、1、31、32、33、257、1000；加入抵消、全负、NaN、+Inf/-Inf 组合。NaN 用 isnan 判断而不是普通差值比较。
4. 用归约构建一行一个 block 的 RMSNorm，测试 rows=0、1、3、128，D=1、31、33、257、1000，gamma 非全 1 与 eps=1e-5。
5. 测归约 n=2^20、2^24；测 RMSNorm [128,256]、[512,1024]、[257,1000]。正确后可选 warp shuffle 版，即线程在 warp 内交换数据，保留共享内存基线。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p10_test 和 p10_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p10_test p10_bench --parallel
mkdir -p reports/results/p10
set -o pipefail
./build/cuda-sm89/bin/p10_test | tee reports/results/p10/tests.txt
./build/cuda-sm89/bin/p10_bench | tee reports/results/p10/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

归约测全阶段合计时间；RMSNorm 输出元素/s，报告最大绝对/相对误差、有限输入容限初设 1e-4+1e-4*abs(reference)，若调整必须有证据。NaN/Inf 按合同单独判断。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 先 memcheck，再 initcheck 检查归约临时结果；共享协作版本另跑 racecheck/synccheck，覆盖非 2 的幂和尾部。
2. 用 ncu 观察共享版与 warp 通信版的同步、调度、资源和有效线程；用 nsys 确认多阶段归约都被计入。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 空输入、非 2 幂、NaN/Inf 规则经过测试。
- RMSNorm 行边界与 gamma 正确，CPU double 参考与 GPU 在容限内。
- 多阶段归约计时不漏后续阶段，报告累积精度与归约顺序影响。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 树形归约默认所有 lane 有有效值：无效 lane 填归约单位元。
- 仅用 abs(NaN-GPU)>eps 会漏报。
- 把 RMSNorm 当减均值的 LayerNorm：本题不减均值。
