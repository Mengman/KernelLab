# P14：CSR 稀疏矩阵乘向量

## 你要学会什么

CSR（压缩稀疏行）用行偏移、列下标和值表示非零元素。实现 SpMV（稀疏矩阵乘向量），观察每行工作量不同对线程与 warp 分工的影响。

## 开始前

前置：P10、P12。阅读 PMPP 中关于稀疏矩阵、CSR 存储、负载不均衡的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float 值与向量，row_ptr 长度 rows+1，首项 0、单调、末项 nnz；col_idx 在 [0,cols)。空行输出 0，rows=0 输出空；重复列项按加法累积。初版索引 int32，实验规模不得溢出。非法 CSR 在 CPU 检查阶段拒绝。

### 创建或修改的文件

- benchmarks/p14/spmv.h：接口与合同。
- benchmarks/p14/spmv_cpu.cpp：CPU reference。
- benchmarks/p14/spmv.cu：GPU baseline 与本题优化版。
- benchmarks/p14/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p14.md 和 reports/results/p14/：报告与原始日志。

## 按步骤完成

1. 写 CSR 校验与 CPU double 累积参考，手算含空行和重复列的一小矩阵。
2. GPU baseline 一线程一行，输出覆盖旧值；验证 0 行、全空行、1×1、3×5、不均匀行。
3. 写一 warp 一行版，用 warp 内归约处理行数据，明确小行和长行的执行方式。
4. 构造均匀每行 8/32/128 非零、90% 短行加少量长行两类矩阵；固定随机种子，输出行长统计。
5. 测 rows=4096、65536，cols=65536，另加入 4097 行的边界样本；相同 CSR 比较两版本，报告每行长度分布而不只给平均稀疏度。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p14_test 和 p14_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p14_test p14_bench --parallel
mkdir -p reports/results/p14
set -o pipefail
./build/cuda-sm89/bin/p14_test | tee reports/results/p14/tests.txt
./build/cuda-sm89/bin/p14_bench | tee reports/results/p14/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

按 2*nnz FLOP 计算 GFLOP/s，列出 ms、nnz/s；行偏移、列下标和向量读取须分开估算，缓存下不能当成精确显存流量。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 对均匀短行与长短混合 CSR 用 ncu 比较有效线程、调度和 L2/DRAM；结合行长分布解释尾部工作，不能只看平均稀疏度。
2. 用 memcheck 检查空行和最后一行；非法 CSR 在 CPU 阶段拒绝，不能把分析工具当作输入校验。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 非法 CSR 被拒绝，空行与重复列正确。
- CPU/GPU 在预先声明的容限内，报告长行最大误差。
- 至少两种行长分布和两种调度实测。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 把 nnz 均匀等同于每行均匀。
- 空行写入旧输出而非 0。
- 只比较总和不逐行比较：可能掩盖行错位。
