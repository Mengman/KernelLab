# GPU Kernel Lab + Tiny Transformer：白话版学习与作业计划

这份文档把两本书变成一条可执行的路线：每一题都先写 CPU 正确版本，再写最直白的 CUDA 版本，最后才做优化。代码在 benchmarks/，实验报告在 reports/，本地题目说明在 docs/assignments/。

## 1. 先把术语说成人话

- host/device：host 是 CPU 和主机内存；device 是 GPU 和显存。
- H2D/D2H/D2D：H2D 是 CPU 到 GPU，D2H 是 GPU 到 CPU，D2D 是 GPU 显存内部复制。
- kernel：在 GPU 上执行的函数。
- thread/block/grid：一个 thread 做一小份工作；多个 thread 组成 block；所有 block 组成 grid。
- warp：通常是 32 个线程一起执行。一个 warp 的线程走不同分支会互相等待。
- global/shared/register：global 容量大但慢；shared 由一个 block 共享；register 是每个线程私有且最快。
- bandwidth：每秒搬运多少数据，通常用 GB/s。
- latency：一次操作从开始到完成的时间。
- FLOP：一次浮点运算。FMA 是 a*b+c，通常算 2 个 FLOP。FMA throughput 是每秒完成的 FMA 数，不是硬件宣传数字，也不是程序必然达到的速度。
- occupancy：SM 同时驻留线程数相对上限的比例；高占用率不等于高性能。
- reduction：把很多数合成一个数，例如求和或最大值。
- scan：前缀和，例如 [2,3,1] 变成 [2,5,6]。
- GEMM：矩阵乘法 C=A×B。
- Tensor Core：GPU 中专门做矩阵乘法的硬件。B200 特有内容单独作为选作题。
- profiler：记录 kernel 用时、内存吞吐和等待原因的工具。

“性能账本”只是一个表格：输入大小、运行次数、耗时、GB/s 或 GFLOP/s、GPU 型号和编译参数。每题都要填它。

## 2. 每道题都这样做

1. 先写 CPU reference，明确边界和误差容限。
2. 写最简单的 GPU baseline，先过正确性测试。
3. 测空输入、很小输入、非对齐尺寸和随机输入。
4. 先预热，再重复测量；报告中位数、最小值和必要的尾延迟。
5. 一次只改一个变量，例如 block 大小或 tile 大小。
6. 用 profiler 或计算量分析解释结果。
7. 把命令、环境、失败尝试和下一步写进报告。

通用构建：

~~~bash
cmake --preset cuda-sm89       # 4090D
cmake --build --preset cuda-sm89 --parallel
ctest --preset cuda-sm89
~~~

GB10 使用 cuda-sm121。没有 CUDA 时先做 CPU reference、接口、测试和报告，不要凭空填写 GPU 数字。

## 3. PMPP 作业

### P01：设备、数据搬运和 FMA

目标：回答“我的 GPU 是什么”“搬一批数据要多久”“简单乘加能跑多快”。

步骤：
1. 运行 device_inspect，保存 GPU 名称、计算能力、显存、SM 数量。
2. 写 c[i]=a[i]*b[i]+c[i] 的 CPU reference。
3. 写 CUDA kernel，使用 grid-stride loop。
4. 分别测 H2D、D2H、D2D；预热 20 次，正式测 100 次，用 CUDA Event。
5. 测 1 MiB、16 MiB、256 MiB；报告中位耗时和有效带宽。
6. 测 FMA kernel，报告 GFLOP/s，并说明这是本实验的实际吞吐。
7. 4090D 和 GB10 分别运行；没有实测的一列写“未测”。

文件：benchmarks/p01/main.cpp、reports/p01.md。完成标准：结果与 CPU 一致，三种规模都有数据，别人能复制命令重跑。常见错误：异步操作未同步、只测一次、把 GB/s 当固定规格、未检查 CUDA 错误。

### P02：一维向量运算

实现 out[i]=alpha*x[i]+beta*y[i]。先朴素索引，再 grid-stride loop。测试 0、1、31、32、33、1000，解释 32 附近的差异。

### P03：二维索引、滤波和 GEMM

实现 3x3 均值滤波和朴素矩阵乘法。明确行列、leading dimension 和边界策略；与 CPU 逐元素比较。

### P04：分支和 block 大小

同一 kernel 测 64/128/256/512 threads per block。记录时间、寄存器、占用率和分支等待；不要只看占用率下结论。

### P05：shared memory 与 tiled GEMM

实现直接转置和 shared-memory tile 转置，再做 tiled GEMM。检查边界，并解释 bank conflict 和合并访问。

### P06：第一张性能表

统一比较 P02–P05。表格必须含输入规模、时间、GB/s、GFLOP/s、GPU、编译选项，并写出原先猜测与实际结果。

### P07：二维卷积

实现多通道卷积的直白版，再用 shared memory 缓存输入 tile。测试 padding、stride 和小尺寸边界。

### P08：stencil

实现 1D 或 2D stencil 的多个时间步。比较每步写回显存与减少中间写回的方案，画数据依赖。

### P09：histogram

先做全局原子直方图，再做每个 block 的局部直方图和合并。用高冲突、低冲突输入解释原子瓶颈。

### P10：归约和 RMSNorm

实现 sum、max、平方和归约，再实现 RMSNorm。处理非 2 的幂、空输入、NaN 和 Inf。

### P11：scan 和压缩

实现 exclusive scan，用它把满足条件的元素写入连续数组。先单 block，再多 block；画出输出位置如何产生。

### P12：并行合并

合并两个有序数组。测试空数组、重复值和极不平衡长度；报告负载如何分配。

### P13：radix sort 和 top-k

实现固定宽度整数 radix sort，再实现 top-k。比较全量排序和局部选择的代价。

### P14：CSR SpMV

实现稀疏矩阵乘向量。改变稀疏度和每行非零元数量，记录负载不均衡。

### P15：BFS

用 frontier 数组实现 breadth-first search。测试链、星形图和随机图，解释邻居数量差异带来的问题。

### P16：Tiny MLP

实现两层 MLP 的 GEMM、bias、ReLU。先串行 kernel，再融合 bias+ReLU，比较中间张量和时间。

### P17：不规则 gather/scatter

用 embedding lookup 或简化 MRI 模拟随机访问。比较直接索引、排序索引和缓存友好布局。

### P18：空间分桶

为粒子建立二维空间桶，再计算邻域。记录桶大小和全局内存访问的影响。

### P19：kernel dispatcher

按输入形状选择小矩阵、长向量、对齐和不对齐路径。为每条路径写回归测试。

### P20：streams

把数据切成多个 chunk，尝试重叠 H2D、kernel、D2H。比较单 stream 和多 stream，并确认硬件确实支持并发。

### P21：dynamic parallelism（选做）

让父 kernel 启动子 kernel，测额外启动开销；不支持时提交设计、伪代码和替代方案。

### P22：CUDA Graph 和 workspace

把固定推理步骤捕获为 CUDA Graph，比较重复 launch 开销；用 workspace 避免频繁申请释放显存。

### P23：总结

选择 P05、P10 或 P16，写 2–4 页：基线、改动、正确性、性能表、瓶颈、失败尝试和下一步。

### PA：数值精度（选做）

比较 FP32、TF32、FP16/BF16 的速度和误差。报告绝对误差、相对误差和最大误差，说明精度损失是否值得。

## 4. Modern GPU Programming for MLSys 作业

- M01：画 CPU、grid、block、warp、线程和内存关系图，用 P02 验证。
- M02：给 P03 算每个元素的字节数和 FLOP，判断内存受限还是计算受限。
- M03：写 layout 模拟器，比较 row-major、column-major 和 swizzle 的地址。
- M04：在 GB10、4090D 测不同 Tensor Core tile，记录形状、类型和误差。
- M05：GB10 探索 TMA；4090D 做 cp.async 或预取对照，说明哪些功能受设备限制。
- M06：实现普通 GEMM 和 Tensor Core GEMM，比较 tile、累加精度和边界处理。
- M07：TMEM（B200 选作）：拿到 B200 后实现；当前只写接口草图和硬件要求。
- M08：用异步 barrier 组织流水线，明确生产者、消费者和同步点。
- M09：用 work queue 模拟 CLC 的“完成一个任务再领取下一个”。
- M10：把算子拆成布局、搬运、计算、同步四层，并写每层不变量。
- M11：为 GEMM/attention 写线程到 tile 的映射表，检查重复读取。
- M12：完成 baseline、shared-memory tiled、Tensor Core 三版 GEMM。
- M13：比较 1-stage、2-stage、3-stage 异步 GEMM。
- M14：尝试 warp specialization 或 persistent kernel，必须保留基线。
- M15：实现 online softmax、causal mask、GQA；B200 上完整 FA4 为选作。

## 5. Tiny Transformer 推理引擎

- T0：定义 Tensor、Shape、DType、Device、Allocator，先让 CPU 路径运行。
- T1：实现小词表 tokenizer 和固定格式权重加载，加入校验。
- T2：实现 embedding、linear、bias、activation、RMSNorm、softmax，每个都有 CPU reference。
- T3：实现 QKV、RoPE、causal mask、KV cache，先做单 token decode。
- T4：逐个把 GEMM、RMSNorm、softmax、attention 迁移到 CUDA，保留对照测试。
- T5：实现 prefill 和 decode，报告每 token 延迟、吞吐、显存和序列长度。
- T6：加入 dispatcher、CUDA Graph、workspace 复用和完整 benchmark；回答哪个 kernel 最慢、为什么、改后改善多少。

每周完成一题或半题：读书、写 CPU、写 CUDA、测试、写报告。遇到缩写先查本文件术语表，不要跳过实验。


## 6. CUDA 开发、调试与性能分析工具

工具练习已经融入 P01–P23 和 PA，详细要求见 [工具路线](assignments/pmpp/TOOLING.md)。先用 Compute Sanitizer 检查 GPU 内存和同步，再用 Nsight Systems/NVTX 找全流程瓶颈，按需用 Nsight Compute 看一个 kernel，结合 ptxas 资源和设备指令提出修改，最后正常测速复验。cuda-gdb 用独立调试配置完成小输入断点与变量检查；主机段错误使用主机调试工具。

Modern GPU 和 Tiny Transformer 阶段延续相同流程：矩阵流水线看共享内存/计算资源，attention 看多阶段流量，prefill/decode 用 NVTX 标出层和 token，streams/Graph 必须用实际时间线验证。未拿到对应硬件或分析权限时写未测，不将另一台 GPU 指标套用。

教学仍由学习者写实现，指导者解释原理和检查证据；不提供完整答案。工具日志、版本、输入、launch、replay/cache 设置和普通 benchmark 结果均可追溯。
