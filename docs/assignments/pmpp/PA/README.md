# PA：数值精度、速度与误差（选做）

## 你要学会什么

比较 FP32、TF32、FP16/BF16 的速度和误差，理解较低精度的适用范围。TF32 是特定计算路径使用的精度格式，不是普通内存数组类型；FP16/BF16 是两种 16 位浮点表示。

## 开始前

前置：P05、P10，建议 M06 已完成。阅读 PMPP 中关于浮点表示、舍入、混合精度与 Tensor Core的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

本题选做。环境不支持时按本题要求交设计和限制说明，不阻塞主线。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

选择 GEMM，CPU double 累积作为参考。保持数学输入相同，先记录从 FP32 转低精度的量化误差，再比较最终结果；低精度路径使用 FP32 累积并明确输入、计算与输出类型。TF32 只在实际支持并启用的矩阵计算路径测试，不声称普通 float kernel 自动是 TF32。

### 创建或修改的文件

- benchmarks/pa/precision.h：接口与合同。
- benchmarks/pa/precision_cpu.cpp：CPU reference。
- benchmarks/pa/precision.cu：GPU baseline 与本题优化版。
- benchmarks/pa/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/pa.md 和 reports/results/pa/：报告与原始日志。

## 按步骤完成

1. 写 double 参考和误差统计：最大绝对误差、最大相对误差（分母 max(abs(reference),1e-8)）、RMSE 即均方根误差、超容限元素数量。
2. 保留 FP32 GEMM；先查设备和使用库/API 的具体支持类型。选择 cuBLAS 或已完成 Tensor Core 实现作为显式低精度路径，不把更换算法的收益全归因于 dtype。
3. 生成 [-1,1] 随机、大小量级混合、抵消、接近 FP16 溢出范围四类有限输入；溢出与非有限输出单独统计而不忽略。
4. 测 (M,N,K)=(128,128,128)、(512,512,512)、(257,511,129)，支持路径处理或明确回退；FP32/TF32/FP16/BF16 不支持的项写未测。
5. 同时列转换成本与纯计算成本，固定预热20/正式100，报告误差-速度表。为指定下游用途提出容限并说明是否满足，不给所有任务通用的值得结论。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 pa_test 和 pa_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target pa_test pa_bench --parallel
mkdir -p reports/results/pa
set -o pipefail
./build/cuda-sm89/bin/pa_test | tee reports/results/pa/tests.txt
./build/cuda-sm89/bin/pa_bench | tee reports/results/pa/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

GEMM 按 2MNK FLOP；性能模式、库版本、Tensor Core 使用条件与累积精度一起记录。纯计算时间和含转换端到端时间分列。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 选做：用 ncu section 和 cuobjdump/SASS 确认 FP32/TF32/FP16/BF16 的真实计算路径与累积精度，dtype 本身不是 Tensor Core 使用证据。
2. 用 nsys/NVTX 分开格式转换和矩阵计算，再用普通 benchmark 得到含转换和纯计算时间；误差统计仍由高精度参考完成。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 参考精度、输入量化、累积和输出精度写清楚。
- 零参考、抵消和溢出测试不漏报。
- 速度比较注明算法/库差异，硬件不支持项未测。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 将 TF32 当成可直接分配的普通数组 dtype。
- 用接近零参考直接除导致相对误差失真。
- 只测随机小值就断言低精度总是安全。
