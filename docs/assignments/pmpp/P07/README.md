# P07：多通道二维卷积

## 你要学会什么

将 P03 单通道滤波扩展到多通道卷积，理解输出形状与输入复用。先实现直白版，再将输入区域缓存到 shared memory；这为后续神经网络算子提供经验。

## 开始前

前置：P03、P05。阅读 PMPP 中关于卷积、输入复用、共享内存 tile 与边缘数据的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float，输入 NCHW 布局即批次/通道/高/宽，权重 OIHW 即输出通道/输入通道/核高/核宽。初版实现互相关：不翻转权重，不加 bias，dilation=1，零填充，stride 可为 1 或 2。核高/宽为 Rh/Rw，步幅为 Sh/Sw，填充为 Ph/Pw。输出 Hout=max(0,floor((H+2Ph-Rh)/Sh)+1)，Wout=max(0,floor((W+2Pw-Rw)/Sw)+1)；用有符号运算和向下取整，避免负数下溢。N=0 或输出空间尺寸为 0 时返回空输出；C、K、Rh、Rw 和步幅必须正，padding 非负。

### 创建或修改的文件

- benchmarks/p07/conv.h：接口与合同。
- benchmarks/p07/conv_cpu.cpp：CPU reference。
- benchmarks/p07/conv.cu：GPU baseline 与本题优化版。
- benchmarks/p07/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p07.md 和 reports/results/p07/：报告与原始日志。

## 按步骤完成

1. 在接口中定义 N,C,H,W,K、kernel_h/kernel_w、padding_h/padding_w、stride_h/stride_w，避免用同一字母指核宽和步幅。写输出形状检查和 CPU reference。
2. GPU 基线每线程一个输出元素；通道和核内循环先保持直接实现。验证 (N,C,H,W,K)=(1,1,3,3,1)、(1,3,7,9,4)，核 1×1、3×3，padding 0/1，stride 1/2。
3. 加入空批次、输出为空、尺寸小于卷积核、非整齐尺寸以及随机输入；非法 stride=0 要报错。
4. 实现共享输入 tile 版，先只支持 3×3、stride=1、padding=1；其他配置显式走基线。tile 外缘需要额外输入，即 halo（为输出边缘提供邻居的额外区域）。
5. 测 (1,3,64,64,16)、(1,16,128,128,32)、(1,32,257,259,64)，同输入比较基线与 tile 版，记录共享内存用量。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p07_test 和 p07_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p07_test p07_bench --parallel
mkdir -p reports/results/p07
set -o pipefail
./build/cuda-sm89/bin/p07_test | tee reports/results/p07/tests.txt
./build/cuda-sm89/bin/p07_bench | tee reports/results/p07/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

按 2*N*K*Hout*Wout*C*Rh*Rw FLOP 计算 GFLOP/s，包含零填充项的约定计算量；Rh/Rw 分别为核高/宽。记录输出形状与总容量。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 共享输入 tile 版的边缘图像运行 memcheck/racecheck/synccheck，检查 halo 载入与同步。
2. 选同一多通道输入，用 ncu 比较基线与 tile 的全局/共享内存流量、寄存器和占用率；以正常性能和访问计数判断复用是否带来收益。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 布局、互相关而非翻转卷积、padding/stride 和空输出语义明确。
- 优化支持范围有回退路径，所有路径与 CPU 一致。
- 报告解释输入复用与 halo 的代价。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- NCHW 与 NHWC 混用：手算一个多通道输入。
- 无符号计算负输出形状：检查核比输入大时的行为。
- 边界线程绕过同步：载入填零但仍参加同步。
