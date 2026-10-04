# P02：一维向量运算与线程索引

## 你要学会什么

实现 out[i] = alpha * x[i] + beta * y[i]，理解元素如何分配给 GPU 线程，比较一线程一个元素与 grid-stride loop（每个线程按整个 grid 的线程总数跨步处理多个元素）。P01 已练习计时，本题重点是接口、索引和边界，为后续矩阵与 Transformer 张量运算打基础。

## 开始前

复习 PMPP 中向量加法和线程索引的章节，按所用版本目录定位。使用 float 数组与标量，n 表示元素数量。x、y、out 互不重叠且各有至少 n 个元素；out 覆盖原值，不读取旧值。n=0 时允许空指针，CPU 函数和 GPU 包装函数直接返回，不启动零个 block。

4090D 使用 cuda-sm89；GB10 当前无法测试，报告写“未测”。本机无 CUDA 时先做 CPU 部分。以下目标名称是待实现目标，完成 CMake 接入后才能运行。

| 文件 | 职责 |
|---|---|
| benchmarks/p02/axpby.h | CPU 与两种 GPU 启动函数声明 |
| benchmarks/p02/axpby_cpu.cpp | CPU reference，不包含 CUDA 头文件 |
| benchmarks/p02/axpby.cu | 两种 kernel 与包装函数 |
| benchmarks/p02/test_axpby.cpp | 独立正确性测试 |
| benchmarks/p02/bench_axpby.cpp | 独立测速程序 |
| benchmarks/CMakeLists.txt | 显式添加已经完成的文件与目标 |
| reports/p02.md | 环境、结果、分析与复现命令 |

先在作业目录实验，不提前移入公共库。复用 gpulab::bench_stats；CUDA_CHECK 可暂时包含 ../p01/cuda_check.h，不要求本题重构公共错误处理。

## 按步骤完成

### 阶段 1：CPU reference

1. 创建 axpby.h 和 axpby_cpu.cpp，自己设计函数声明，包含两个只读输入数组、输出数组、n、alpha、beta。
2. 用普通循环实现公式，保证输出不依赖其旧值。
3. 创建 test_axpby.cpp，先覆盖空输入、单元素、三元素手算例子。输出预先填入与期望不同的值，验证被覆盖。
4. 测试 alpha、beta 为 0、1、负数的组合；至少包含负输入。
5. CMake 先接入 CPU 库与 p02_test_axpby，使 cpu-debug 可构建运行。GPU 文件完成后才加入 CUDA 条件。

完成这一阶段后由指导者检查，再推进 GPU 部分。

### 阶段 2：一线程一个元素

1. 实现 GPU baseline（基线版本，即用于对照的初始实现），每线程处理一个全局下标，越界线程不访问数组。
2. 每 block 固定 32 线程，自己推导覆盖 n 个元素的 block 数并解释向上取整。
3. 包装函数处理 n=0，启动并检查启动错误；测试调用者负责同步和执行错误检查。
4. 测试 n=0、1、31、32、33、1000，每种长度使用固定和随机输入。
5. 随机输入固定种子、范围 [-10,10]；逐元素比较 CPU/GPU。初始容限为 abs_error <= 1e-5 + 1e-5 * abs(reference)。失败打印长度、下标、CPU/GPU 数值并返回非零退出码。有限数为本题输入范围。

### 阶段 3：grid-stride loop

1. 添加第二个 kernel，保留基线版本用于对照。
2. 固定 32 threads/block、2 blocks，自己推导起始下标和步长；用 n=1000 验证线程少于元素时仍覆盖完整数组。
3. 两种实现运行同一组正确性测试。
4. 记录 n=31、32、33 的有效线程、越界线程与 block 数，解释基线在 33 时为何需要第二个 block。差异来自启动配置，不能只凭耗时断言 warp 本身变慢。

### 阶段 4：性能实验

1. 使用 CUDA Event（GPU stream 上的计时事件），预热 20 次、正式测量 100 次。start、kernel、stop 在同一 stream（有序任务队列），等待 stop 完成再读耗时。分配、初始化、复制与验证在计时外。
2. 小输入测 31、32、33，每 block 32 线程；基线 grid 按 n 计算，grid-stride 固定 2 blocks。差异不明显也是有效结论，不要求 32 附近必有性能突变。
3. 大输入按每个数组 1、16、256 MiB 测量，除以 sizeof(float) 得到 n。两种实现都用 256 threads/block；基线覆盖全部元素，grid-stride 固定 256 blocks。这是在比较两种启动策略，不能将全部差异归因于循环语法。
4. 使用相同输入与非零标量，例如 alpha=1.5、beta=-0.5。每次覆盖 out，无需恢复输出。
5. 输出最小耗时、中位耗时、GB/s、GFLOP/s 和启动配置。沿用现有统计函数并说明 nearest-rank 中位数口径。
6. 先写性能预测，再记录实测，解释固定开销、并行度和缓存的可能影响。

计算公式：

- MiB = 2^20 bytes，GB = 10^9 bytes。
- 两次乘法加一次加法，按 3 FLOP/元素计算，即使编译器使用融合乘加指令仍保持此口径。
- 算法访问量为读取 x、y 和写入 out，共 3 × n × sizeof(float) 字节。
- GFLOP/s = 3 × n / (median_ms × 10^6)。
- 有效访问带宽 GB/s = 3 × n × sizeof(float) / (median_ms × 10^6)。它不是硬件计数器测得的显存流量。

## 构建与运行

完成相应目标后，在项目根目录执行：

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug --target p02_test_axpby --parallel
./build/cpu-debug/bin/p02_test_axpby

cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --target p02_test_axpby p02_bench_axpby --parallel
mkdir -p reports/results
set -o pipefail
./build/cuda-sm89/bin/p02_test_axpby | tee reports/results/p02-tests.txt
./build/cuda-sm89/bin/p02_bench_axpby | tee reports/results/p02-bench.txt
```

CPU-only 只运行 CPU 测试，CUDA 构建加入 GPU 测试，不能静默把未执行的 GPU 测试算作通过。将正确性测试接入 CTest；CPU 测试不依赖 GPU，GPU 测试仅在 GPU 测试开关启用时注册。指导者会协助检查 CMake 组织。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 使用 memcheck 检查两个 GPU 版本在 n=0、1、31、32、33、1000 的行为，保存日志和退出码。
2. 在独立故障样例中引入一个 GPU 越界，读出源码行和 block/thread，再修复复查；故障样例不加入正常构建。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- CPU 和两种 GPU 实现完成，接口与边界明确，输出覆盖旧值。
- 0、1、31、32、33、1000 的固定和随机输入通过。
- 测试与测速独立，CPU-only 不包含 CUDA 头文件。
- 有小输入对照和三种大输入实测，报告含 GPU、计算能力、驱动、Toolkit、编译器、数据类型、输入规模、启动配置、预热、迭代、计时和统计方法。
- 报告写明公式、可复现命令、预测与实际结果；GB10 未运行写“未测”。
- 能解释 31、32、33 的线程分配，不把噪声当作确定规律。

## 常见问题

- n 是元素数，不是字节数；每数组字节数为 n × sizeof(float)。
- n=33 漏最后一个元素：检查 grid 向上取整与边界条件。
- 多次执行结果变化：应覆盖 out，不能累加旧 out。
- CPU-only 找不到 CUDA 头文件：检查公共接口与 CPU 实现的依赖。
- 小输入耗时波动：重复测量并如实记录，不预设 32 边界必有性能跳变。

## 教学方式

学习者编写代码，指导者分阶段解释原理、给 API 和验收标准、检查代码与结果。默认不提供完整实现或直接代写；遇到问题先给原因与局部提示。当前阶段完成后再推进下一阶段，一次改变一个实验变量。
