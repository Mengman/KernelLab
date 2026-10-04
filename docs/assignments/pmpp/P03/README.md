# P03：二维索引、均值滤波和朴素矩阵乘法

## 你要学会什么

把一维索引扩展到二维，学会将行列坐标映射到数组地址。先用滤波掌握边界，再写矩阵乘法，为后续分块和线性层准备基线。

## 开始前

前置：P02。阅读 PMPP 中关于二维线程组织、行主序数组、朴素矩阵乘法的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

使用 float、行主序。图像为 H×W，行跨度 ld>=W；3×3 均值滤波采用零填充且始终除以 9，输出与输入不重叠。矩阵 A 为 M×K、B 为 K×N、C 为 M×N，各行跨度显式传入。滤波 H 或 W 为 0 时不访问数据；GEMM 的 M 或 N 为 0 时返回，K=0 时 C 填 0。

### 创建或修改的文件

- benchmarks/p03/matrix.h：接口与合同。
- benchmarks/p03/matrix_cpu.cpp：CPU reference。
- benchmarks/p03/matrix.cu：GPU baseline 与本题优化版。
- benchmarks/p03/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p03.md 和 reports/results/p03/：报告与原始日志。

## 按步骤完成

1. 在 matrix.h 明确接口和各个行跨度；先手算 2×3 图像的地址及一个 2×3 乘 3×2 的结果。
2. 在 matrix_cpu.cpp 分别实现滤波与 GEMM 的 CPU reference；输出覆盖旧值。测试带额外行填充的输入，保证不把跨度当列数。
3. 在 matrix.cu 写两种朴素 kernel，每线程负责一个输出；先固定 16×16 threads/block，自己推导二维 grid 和边界判断。
4. 验证图像 0×0、1×1、2×3、17×19，以及 GEMM (M,N,K)=(0,3,2)、(2,3,0)、(1,1,1)、(17,19,13)；固定输入、负数、随机数都要覆盖。
5. 测滤波 256×256、1024×1024、2048×2048；测 GEMM (128,128,128)、(512,512,512)、(257,511,129)。计时外验证输出，记录 CPU/GPU 最大误差。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p03_test 和 p03_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p03_test p03_bench --parallel
mkdir -p reports/results/p03
set -o pipefail
./build/cuda-sm89/bin/p03_test | tee reports/results/p03/tests.txt
./build/cuda-sm89/bin/p03_bench | tee reports/results/p03/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

滤波报告 ms、输出像素/s，理论参考按每输出 9 次读取和 1 次写入说明，但不当作实际显存流量。GEMM 按 2MNK FLOP 计算 GFLOP/s，K=0 不作为性能样本。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 对一个小 GEMM 或滤波配置采集 nsys 时间线，为 init/warmup/measure/validate 添加 NVTX 范围，解释 CPU 范围与异步 GPU 工作的关系。
2. 用独立设备 Debug 构建在 cuda-gdb 设置 kernel 断点，查看一个有效线程与一个越界线程的索引；Release 下重新验证。工具不可用先写断点和变量检查计划。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 零填充、固定除以 9 和矩阵行跨度均有测试。
- 两个 CPU reference 与两个 GPU 基线通过小尺寸和非整齐尺寸测试。
- 报告包含两个任务的独立结果，并画出二维坐标到线性地址的映射。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 把 W 与 ld 混用：给每行添加填充元素检测。
- 边界像素除以有效邻居数：那是另一种算法，本题固定除以 9。
- 混淆 grid 的 x/y 与行列：在 2×3 小输入上检查。
