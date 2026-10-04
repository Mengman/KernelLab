# P16：两层 Tiny MLP 与逐元素融合

## 你要学会什么

MLP（多层感知机）把矩阵乘法和逐元素操作组合起来。先做两层网络，再融合 bias 与 ReLU，观察 launch 和中间张量流量的变化，为 Transformer 前馈层做准备。

## 开始前

前置：P05、P10。阅读 PMPP 中关于GEMM、bias、激活函数、算子融合的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float 行主序，Y=ReLU(XW1+b1)W2+b2，不在第二层后添加 ReLU。X=[B,Din]、W1=[Din,H]、W2=[H,Dout]，bias 按列广播。B=0 输出空，Din/H/Dout 必须正；输入权重输出不重叠。

### 创建或修改的文件

- benchmarks/p16/mlp.h：接口与合同。
- benchmarks/p16/mlp_cpu.cpp：CPU reference。
- benchmarks/p16/mlp.cu：GPU baseline 与本题优化版。
- benchmarks/p16/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p16.md 和 reports/results/p16/：报告与原始日志。

## 按步骤完成

1. 写 CPU 全链路参考和每阶段可检查输出；测试 B=1、3，Din/H/Dout 含 1、17、33。
2. 复用 P05 GEMM，先分别 launch bias 和 ReLU，保留中间张量；验证固定和随机权重以及负激活。
3. 将第一层 bias+ReLU 融合成一个逐元素 kernel，第二层 bias 保持独立，其他部分不变。
4. 比较未融合和融合版完整前向时间，先仅一个变量；可选将 GEMM 输出处理融合到 GEMM，但作为额外版本。
5. 测 (B,Din,H,Dout)=(1,128,256,64)、(32,256,512,128)、(127,129,257,65)，每次输入相同且输出覆盖；列出峰值临时显存和 launch 次数。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p16_test 和 p16_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p16_test p16_bench --parallel
mkdir -p reports/results/p16
set -o pipefail
./build/cuda-sm89/bin/p16_test | tee reports/results/p16/tests.txt
./build/cuda-sm89/bin/p16_bench | tee reports/results/p16/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告全前向 ms、样本/s，另列主要 GEMM 约定 FLOP=2*B*Din*H+2*B*H*Dout；bias/激活不直接当 GEMM FLOP。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 给各层 GEMM、bias、ReLU 添加 NVTX；用 nsys 比较融合前后的 launch 数、阶段时间与空隙，保存两张同尺度截图。
2. 对最耗时 GEMM 用 ncu 验证计算/访存限制，再用普通 benchmark 看整个 MLP 是否改善；memcheck 检查非整齐 hidden 大小。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 每层及全链路与 CPU 一致，报告误差累计。
- 优化版仅融合规定阶段，数据和 GEMM 配置不变。
- 记录 launch、流量估算、临时显存及端到端性能。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 误把第二层也加 ReLU，改变网络。
- bias 广播沿行而非列。
- 只测融合 kernel，不测全前向。
