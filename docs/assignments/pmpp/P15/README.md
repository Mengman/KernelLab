# P15：Frontier BFS 图遍历

## 你要学会什么

BFS（广度优先搜索）按层寻找源点到其他点的最少边数。frontier 是当前层待扩展的顶点集合，通过此题练习不规则邻居访问和重复发现的并发处理。

## 开始前

前置：P09、P11、P14。阅读 PMPP 中关于图 CSR、广度优先遍历、并发队列与原子访问的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

有向无权图采用 CSR 邻接表，distance 为 int32，未访问为 -1，源为 0；输出距离，不要求唯一父节点。V=0 允许无源且返回空，V>0 时 source 必须有效。允许自环、重边和不连通图，所有邻居下标先校验。

### 创建或修改的文件

- benchmarks/p15/bfs.h：接口与合同。
- benchmarks/p15/bfs_cpu.cpp：CPU reference。
- benchmarks/p15/bfs.cu：GPU baseline 与本题优化版。
- benchmarks/p15/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p15.md 和 reports/results/p15/：报告与原始日志。

## 按步骤完成

1. 写 CPU 队列 BFS，手算一张 6 顶点图，说明每层 frontier。
2. GPU baseline 由 CPU 逐层发起 kernel，每线程扩展一个 frontier 顶点；用原子操作确保顶点只首次发现一次，下一层容量 V。
3. 验证空图、单点、链、星形、有环、不连通、重边与自环，逐顶点比较距离；别用 frontier 内部顺序判断正确性。
4. 优化高出度任务分配：先选一 warp 扩展一个点或边粒度分工，保留基线。计时包含所有层 kernel 与维持层推进所需同步。
5. 测链图、星形图和固定种子随机图 V=4096、65536，随机平均出度 4/16；记录层数、每层 frontier 大小、访问的边数。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p15_test 和 p15_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p15_test p15_bench --parallel
mkdir -p reports/results/p15
set -o pipefail
./build/cuda-sm89/bin/p15_test | tee reports/results/p15/tests.txt
./build/cuda-sm89/bin/p15_bench | tee reports/results/p15/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告整个 BFS 时间及实际扫描边数/时间（TEPS，即每秒遍历的边数）；不把未访问子图的所有边计入吞吐。另列构图与 H2D 时间，不混入算法时间。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. NVTX 标注每层 frontier，nsys 查看层间 CPU 读计数/同步与 GPU 空隙，分别分析链图和星形图。
2. ncu 采高出度层的代表 kernel，结合原子/访存与每层规模说明负载；Sanitizer 无报错不证明全局首次访问原子协议正确。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- 距离逐点与 CPU 一致，不同 frontier 顺序允许。
- 容量不会溢出，重复邻居不重复入队。
- 链、星形和随机图都测，解释层数与高出度差异。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 普通写 distance 代替原子首次发现：重复入队。
- 每层未清零下一 frontier 计数。
- 只测星形图得到一层极快结果却推广全部图。
