# P08：多时间步 stencil

## 你要学会什么

stencil 是根据邻近元素更新当前位置的计算。本题用一维三点更新观察时间步之间的数据依赖，比较逐步写回显存与在块内复用多个时间步。

## 开始前

前置：P05、P07。阅读 PMPP 中关于stencil、邻域依赖与时间分块的主题，按所用版本目录定位，不假定章节号一致。遵循 [统一实验与教学规则](../RULES.md)。

4090D 使用 cuda-sm89，GB10 未运行写“未测”。本机没有 CUDA 时先做接口、CPU reference、手算测试和报告草稿；GPU 部分在服务器验证。工具不可用如实标记，性能结论可用计算量/访存分析辅助，不伪造 profiler 数据。

### 输入输出与边界

float 一维数组，每步 next[i]=0.25*prev[i-1]+0.5*prev[i]+0.25*prev[i+1]，越界邻居视为 0，边界位置也按此式更新。输入输出分开，不能原地覆盖；T=0 返回输入副本，n=0 返回空结果。

### 创建或修改的文件

- benchmarks/p08/stencil.h：接口与合同。
- benchmarks/p08/stencil_cpu.cpp：CPU reference。
- benchmarks/p08/stencil.cu：GPU baseline 与本题优化版。
- benchmarks/p08/test.cpp、bench.cpp：独立测试与测速入口。
- benchmarks/CMakeLists.txt、tests/CMakeLists.txt：只显式加入实际完成的文件与测试，复用统计库。
- reports/p08.md 和 reports/results/p08/：报告与原始日志。

## 按步骤完成

1. 画 n=5、T=2 的依赖图，写双缓冲 CPU reference；明确最后结果位于哪个缓冲区。
2. 写每时间步一个 kernel 的 GPU 基线，使用两个显存缓冲区轮换；测试 n=0、1、2、31、33、1000，T=0、1、2、5。
3. 实现先固定融合两步的版本：每个输出 tile 载入更宽 halo，在共享内存双缓冲中更新，只写最终内部区域。不要依赖块间同步。
4. 验证 n=257、1003 和随机输入；各 tile 的输出区域不得重叠，不能把其他 block 尚未完成的数据当成新时间步输入。
5. 测 n=2^20、2^24，T=2、20、100；计时整段演化，初值恢复放在计时之外。可选再比较融合 4 步。

先完成第一步交由指导者检查，再推进后续；本题不提供完整实现。保留基线，每次优化只改变一个变量。

### 运行与保存输出

以下是计划目标 p08_test 和 p08_bench 的复现命令；对应文件和 CMake target 实现后才能运行，当前不自动加入构建。CPU 阶段先用 cpu-debug 构建 CPU 测试，GPU 阶段独立启用 CUDA 和 GPU 测试。

```bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p08_test p08_bench --parallel
mkdir -p reports/results/p08
set -o pipefail
./build/cuda-sm89/bin/p08_test | tee reports/results/p08/tests.txt
./build/cuda-sm89/bin/p08_bench | tee reports/results/p08/bench.txt
```

### 计时与指标

默认预热 20 次、正式测量 100 次，记录最小和中位耗时，统计方式沿用 nearest-rank。GPU 用 CUDA Event；主机工作或完整流水线需要另列 CPU 单调时钟结果。执行前恢复的状态、哪些阶段计时及哪些阶段排除，必须写入报告。

报告整段 ms、元素更新/s=n*T/时间，并写明边界与 halo 重算未计入该有效更新量。可估算基线流量，但不能把融合版等同于基线字节数。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 对融合两步版本的尾 tile 运行 Sanitizer，特别检查共享内存双缓冲和同步；记录工具未覆盖的块间依赖证明。
2. 用 NVTX 标注各时间步/融合组，在 nsys 统计 launch 数，用 ncu 采代表 kernel，区分 halo 重算和全局流量减少的证据。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- T=0 和奇偶 T 均正确，CPU/GPU 使用同一边界。
- 两步融合版验证输出所有位置且无块间依赖错误。
- 报告依赖图、halo 大小、全局流量估算与性能对照。
- 有独立正确性测试、完整环境与命令、原始日志；报告区分实测、规格和推测。未运行的设备或工具明确注明。

## 常见问题

- 原地更新导致读取已更新邻居：使用双缓冲。
- 漏掉 halo 随时间步扩展：从依赖图推导。
- 计时只覆盖一个 kernel：必须测完整 T 步。
