# P17：Embedding gather 与重复索引 scatter

## 你要学会什么

gather 按索引读取元素，scatter 按索引写入元素。以 embedding lookup（按 token ID 读取向量）和 scatter-add 为例，理解随机访问、重复地址与数据重排的收益及代价。

## 开始前

前置：P09、P13、P16。阅读 PMPP 中关于不规则访存、gather/scatter、索引重排的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float embedding [V,D] 与 int32 token IDs [N]，输出 [N,D] 保持原 token 顺序；ID 越界拒绝。scatter-add 将 [N,D] 更新加到清零的 [V,D]，重复 ID 累加。N=0 返回空 gather 和全零 scatter；V,D 必须正。

### 创建或修改的文件

- benchmarks/p17/gather_scatter.h：接口与合同。
- benchmarks/p17/gather_scatter_cpu.cpp：CPU reference。
- benchmarks/p17/gather_scatter.cu：GPU baseline 与本题优化版。
- benchmarks/p17/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p17.md 和 reports/results/p17/：报告与原始日志。

## 按步骤完成

1. 写 CPU gather 和 double 累积 scatter-add，测试重复 ID、首尾 ID、非法 -1/V、N=0。
2. GPU baseline 直接索引 gather，scatter-add 用原子更新；明确每线程处理的维度和 token。
3. 实现按 ID 排序再访问的版本，携带原位置恢复 gather 顺序；scatter 可先合并相同 ID 再写出。
4. 比较访问阶段和包含排序/恢复的全流程时间，不能只报告重排后 kernel。可选复用预先排序索引，另列摊销条件。
5. 测 V=65536，D=32/128，N=4096/65536；输入分布为顺序、均匀随机、90% 重复热门 ID。scatter 每轮清零计时口径固定。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p17_test 和 p17_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p17_test p17_bench --parallel
mkdir -p reports/results/p17
set -o pipefail
./build/cuda-sm89/bin/p17_test | tee reports/results/p17/tests.txt
./build/cuda-sm89/bin/p17_bench | tee reports/results/p17/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告 gather/scatter 各自 ms、token/s、维度和索引分布；gather 可按 2*N*D*4 给有效读写量，索引额外读取需说明。scatter 重复写流量不能简单等同于 gather。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 用 ncu 对顺序、随机、热门 ID 的访问比较 L2 命中和 DRAM 流量；说明索引重排是否只是改变缓存。
2. nsys/NVTX 标出排序、恢复顺序和实际 gather/scatter，完整时间包含预处理；用 memcheck 检查合法首尾 ID，错误 ID 在 launch 前拒绝。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- gather 恢复原顺序，重复 ID scatter 结果在容限内。
- 非法 ID 在 launch 前检查，输入合同明确。
- 重排优化包含排序代价，同时列复用索引情形。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 排序 ID 后忘恢复输出顺序。
- scatter 每轮在旧数据上累加。
- 原子浮点顺序不同要求逐 bit 相等。
