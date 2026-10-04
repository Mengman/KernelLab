# P19：按输入条件选择 kernel 的 Dispatcher

## 你要学会什么

dispatcher 是根据输入条件选择具体 kernel 的入口。复用已测算子，按形状和对齐选路径，练习将性能实验转成可验证的选择规则，为推理引擎准备调度层。

## 开始前

前置：P05、P06、P16。阅读 PMPP 中关于算法选择、接口合同与回归测试的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

初版选择 GEMM：同一行主序 float 接口，输出和边界沿用 P03；所有被选择版本必须支持相同数学合同。选择规则只能检查已知 shape、跨度和实际指针对齐，不能只凭维度猜地址对齐。

### 创建或修改的文件

- benchmarks/p19/dispatch.h：接口与合同。
- benchmarks/p19/dispatch_cpu.cpp：CPU reference。
- benchmarks/p19/dispatch.cu：GPU baseline 与本题优化版。
- benchmarks/p19/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p19.md 和 reports/results/p19/：报告与原始日志。

## 按步骤完成

1. 整理朴素、tile=16、tile=32 等实际完成版本的支持范围，写明零尺寸和未对齐输入回退路径。
2. 定义 dispatcher 入口与可查询的路径名，先用明确规则选择，不做复杂注册框架或启动时自动搜索。
3. 同一组输入直接调用各版本和 dispatcher，验证数学结果与路径选择；每条路径至少一个命中与一个拒绝样本。
4. 测试阈值前后形状、偏移一个 float 的未对齐指针、带跨度、零尺寸；不支持的优化必须显式回退。
5. 在小 GEMM (16,16,16)、非整齐 (17,33,19)、中大 (256,256,256)/(1024,1024,1024) 上测强制各路径与自动选择；用实测证据决定阈值，留一组未参与调参输入验证。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p19_test 和 p19_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p19_test p19_bench --parallel
mkdir -p reports/results/p19
set -o pipefail
./build/cuda-sm89/bin/p19_test | tee reports/results/p19/tests.txt
./build/cuda-sm89/bin/p19_bench | tee reports/results/p19/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告路径名、条件、kernel ms 和主机选择开销；CPU 选择用稳态 CPU 时钟测，不用 GPU Event 计 CPU 判断。自动选择不要求每形状全局最优，但不能选择不合法路径。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. NVTX 标记实际选择路径，nsys 核对自动选择的 kernel 与程序输出路径名一致，重点看阈值与未对齐回退。
2. ncu 对同一 shape 的强制路径和自动路径采样；CPU 选择时间单独测，不从 GPU Event 或 NVTX 区间直接推断 dispatch 净开销。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 所有路径的合同一致，回退测试覆盖边界和未对齐。
- 选择规则有实测依据，未参与调参样本通过。
- CPU-only 调度接口不泄露 CUDA 依赖，报告选择失败或无收益情形。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 只检测 n 的倍数却忽略指针偏移。
- 阈值来自某台 GPU却默认为所有 GPU 最优。
- 优化没覆盖边界而 dispatcher 仍选择它。
