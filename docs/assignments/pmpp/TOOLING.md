# CUDA 调试与性能分析工具练习

## 你要学会什么

把“程序正确”“哪个阶段耗时”“某个 kernel 为什么慢”分开验证，再根据证据改代码。工具练习随 P01–P23 推进，不要求先安装所有工具，不用 profiler 输出替代正常 benchmark，也不由指导者代写 kernel。

本路线以命令行在 Linux GPU 服务器采集，报告可下载到本机 GUI（图形界面）查看；本机没有 GPU 仍可阅读采集好的报告。未确认工具、权限或硬件能力时如实标注，不假装完成。

## 工具分别解决什么问题

| 工具 | 需要回答的问题 | 典型输出与限制 |
|---|---|---|
| CUDA Event 与 CPU 单调时钟 | 不带分析工具时到底多快？ | 保留原有 20 次预热、100 次正式测量与 min/median，作为性能主结果 |
| Compute Sanitizer | GPU 是否越界、读未初始化显存或错误同步？ | memcheck/initcheck/racecheck/synccheck 日志；不保证发现全部错误 |
| Nsight Systems，CLI 为 nsys | CPU 提交、复制、kernel、同步如何排列？哪里有等待和空隙？ | .nsys-rep 时间线与 stats；需要有实际重叠证据才能声称并发 |
| Nsight Compute，CLI 为 ncu | 单个 kernel 的资源、访存和执行效率如何？ | .ncu-rep、section、指标；通常有 replay（为不同计数器重复执行），会改变运行条件 |
| NVTX（NVIDIA Tools Extension） | 时间线上哪些操作属于预热、计算或某一层？ | 程序主动标注名称和区间；标注本身不是 GPU 同步 |
| nvcc / ptxas 资源报告 | 编译后每线程用了多少寄存器？是否出现 local memory 流量？ | 构建日志中的寄存器、共享内存、栈和 spill 信息 |
| cuobjdump / nvdisasm | 源码最终生成了哪些设备指令？ | PTX（中间指令）与 SASS（GPU 机器指令）；不能只凭源码判断融合和分支 |
| cuda-gdb | GPU 出错时是哪条语句、哪个线程/块？ | 断点、变量、线程状态；调试构建不可用于性能结论 |
| 主机 gdb / AddressSanitizer（选做） | CPU 端越界或段错误发生在哪里？ | CPU backtrace/内存诊断；不代替 GPU Sanitizer |

先用 Sanitizer 过正确性，再用 nsys 找整段瓶颈，必要时用 ncu 看具体 kernel，最后脱离 profiler 正常复测。不是每道题都要收集所有指标。

## 开始前：版本、路径和权限

在 GPU 服务器的项目根目录运行并保存工具清单：

```bash
mkdir -p reports/results/tooling
command -v nsys ncu compute-sanitizer cuda-gdb cuobjdump nvdisasm
```

分别对实际存在的工具运行 `--version`，保存版本输出；检查 `nvcc --version`、`nvidia-smi`，记录 GPU 与驱动。CLI 不在 PATH 不代表未安装：查管理员提供的 Toolkit/Nsight 安装目录，再使用确定存在的完整路径。不要全盘扫描或照猜测路径执行。

2026-10-05 已恢复 `ssh 4090` 连接，实查 `nvcc` 为 `/opt/conda/bin/nvcc`，版本 12.6.85。PATH 和已检查的 `/opt/conda`、`/usr/local`、`/usr/lib` 安装目录（查找深度至 5）未找到 nsys、ncu、compute-sanitizer、cuda-gdb、cuobjdump、nvdisasm；这次无法做实际工具采集，其他未检查位置是否有安装仍未知。Conda 清单包含 cuda-nvtx/cuda-nvtx-dev 与 nvcc/ptxas 相关包；有 CUPTI 或 CUDA 编译器不等于已安装 Nsight。GPU 计数器权限尚未测试。检查原始输出保存到 `reports/results/tooling/environment.txt`。

后续先完成工具环境准备：按 Ubuntu/Conda 实际环境与驱动兼容性选择 NVIDIA 官方工具发行包，优先仅补充所需工具，不为学 profiler 更换整个 Toolkit。安装完成再验证版本、最小采集与计数器权限；未安装不填写成功结果。此轮只添加教学文档和保存环境证据，没有安装软件或修改驱动配置，也不改变已完成的 P01 状态。

遇到 `ERR_NVGPUCTRPERM` 表示 GPU 性能计数器权限不足；容器里的 root 也未必具有宿主机权限。保存错误和尝试命令，由服务器管理员配置，不能把失败或空报告写成通过，不要求随意改驱动安全设置。CPU sampling 被限制时，可先采用关闭采样的 CUDA 时间线采集方式。工具不可用时完成分析设计与算法估算，硬件证据栏写未测。

官方文档会更新，实际选项以本机 `--help`、版本对应文档为准。Nsight GUI 与采集 CLI 的报告兼容性需核对；安装 GUI/升级 CLI 是环境任务，不因布置练习自动执行。

## 编译方式：分析和调试分开

性能分析使用 Release 优化和 CUDA `-lineinfo`，它用于把设备指令对应到源码行；给实际包含 .cu 的 target 加该选项，而不是只给 .cpp 可执行 target。可选 `--ptxas-options=-v` 获取编译资源报告，保存一次重新编译该 target 的日志。

cuda-gdb 源码调试另用独立构建目录，按所用版本启用主机调试信息和 CUDA `-G` 设备调试。`-G` 通常改变优化与代码生成，因此不能拿 Debug 的时间和原 Release 结果比较。仅 CMAKE_BUILD_TYPE=Debug 不一定等于设备代码已经启用所需调试选项，检查实际 nvcc 命令。

不要覆盖正常测速配置，也不在整个项目永久添加调试和 profiler 参数。Compute Sanitizer 支持对优化程序检查，可先用 Release + lineinfo 的小测试复现。编译选项依据：[NVCC 官方说明](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)。

## 准备一个小的分析工作负载

普通 benchmark 继续按 20/100 运行。另增加由你实现的分析入口或参数：一次只选择一个算法、一个 shape、一个 grid/block，20 次预热后仅执行 1–5 次目标工作，并在末尾验证结果。输出具体配置，保存与普通 benchmark 相同的随机种子。

这些参数还没有写入所有程序，不能复制一个不存在的 `--profile` 参数直接运行。先在 P03 或 P04 实现独立分析模式，指导者审阅接口后再采集。每题的附加练习写明需要选择的配置。

多阶段算法的分析入口应保留完整的阶段和依赖。ncu 的 `--launch-skip` 是跳过符合过滤器的 launch，不是自动跳过全程序的预热；只有确定过滤到的目标 kernel 在一个配置内预热了 20 次，才能使用数值 20。扫描多配置时应分次运行，不把第一个小输入的结果当作最后一个大输入。

## 工具练习 A：Compute Sanitizer 与错误定位

### 基础：P02

1. CPU/GPU 数值测试先通过，再对空输入、31/32/33、1000 和不整齐大输入运行 memcheck。避免把完整性能 sweep 放进 Sanitizer。
2. 保存实际检查日志与退出码，报告检查了哪一实现、哪些输入、工具版本。
3. 在独立且不参与正常构建的调试样例中，自己引入一次 GPU 越界写，观察错误下标、block/thread、源码行；修复后复跑并保留前后日志。不要把故障注入到正常 benchmark 中。

P01 已有目标可用作命令练习，先确认工具已安装：

```bash
mkdir -p reports/results/tooling
set -o pipefail
compute-sanitizer --tool memcheck --error-exitcode 1 \
  ./build/cuda-sm89/bin/p01_test_fma 2>&1 \
  | tee reports/results/tooling/p01-memcheck.txt
```

### 进阶：P05、P08、P10、P11

对含共享内存和 barrier 的小输入分别运行 racecheck 与 synccheck。racecheck 主要检查共享内存访问危害，不能证明全局原子队列或所有 stream 间依赖没有竞态；synccheck 检查同步使用，不能证明算法结果正确。

initcheck 主要检查未初始化全局显存读取（具体新功能看所用版本）。先过 memcheck，再运行 initcheck；不要默认它检查 CPU 初始化或全部共享内存初始化。P10/P11 重点检查尾部和临时数组。

只将已有小测试程序路径赋给 `test_app`，以下是可以复用的命令框架：

```bash
test_app=./build/cuda-sm89/bin/p01_test_fma
compute-sanitizer --tool initcheck --error-exitcode 1 "$test_app"
compute-sanitizer --tool racecheck --error-exitcode 1 "$test_app"
compute-sanitizer --tool synccheck --error-exitcode 1 "$test_app"
```

P01 没有共享内存协作，因此无告警不能代替 P05/P10 上的检查。练习应在对应算法实现后替换实际路径。资源受限或执行很慢时缩小输入，不跳过关键边界。

官方依据：[Compute Sanitizer 工具说明](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

## 工具练习 B：Nsight Systems 看整个程序

### 基础：P01 回顾、P03

1. 对一个小工作负载采集 CUDA 时间线，找到 CPU Runtime API、GPU kernel 和复制所在轨道。
2. 区分 CPU 上 API 调用时间与 GPU 上实际工作时间；一次 API 返回不代表异步工作已完成。
3. 找到预热、正式测量和验证，说明 Event 同步/设备同步发生在哪里。区分初始化空隙与稳态空隙。

下面命令使用已存在的 P01 FMA 程序；它会采集多个配置，适合首次认识时间线。后续用单配置分析入口缩小范围：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o reports/results/tooling/p01-timeline \
  ./build/cuda-sm89/bin/p01_bench_fma
nsys stats reports/results/tooling/p01-timeline.nsys-rep \
  > reports/results/tooling/p01-timeline-stats.txt
```

`--sample=none` 与 `--cpuctxsw=none` 关闭 CPU 采样/线程调度采集，初次只看 CUDA 与标注。程序未添加 NVTX 时不会凭空出现用户标注。重复采集换输出名；没有报告文件先检查错误，不直接运行 stats。

学会从 GUI 打开 .nsys-rep：时间放大到一个正式迭代，标出 H2D 恢复 c、kernel、等待和下一轮。H2D 不在 Event 区间仍可能改变缓存和整段耗时。报告截图应有轨道名、时间刻度和输入配置，不只截一个不明百分比。

### 进阶：P16、P20、P22

- P16：记录未融合/融合后的 launch 数和层间空隙，不凭单 kernel 时长判断全前向收益。
- P20：画单 stream 和多 stream 时间线，确认传输与计算确有重叠，检查复用槽位导致的等待。
- P22：分别标记 workspace 初始化、Graph 创建/实例化、首次 replay 和稳态 replay。必要时按版本选择 Graph node 级跟踪，注明其开销，不能把一次 graph-level 记录误当成一个实际 kernel。

正式 profile 模式可在预热后使用 `cudaProfilerStart/Stop` 限定采集区域；先在停止前确认目标异步工作完成。nsys 需要显式启用 `--capture-range=cudaProfilerApi`；没有代码调用 start 时可能得到空采集。首次先采全流程理解再缩范围，不要求一开始就把所有 capture 选项配齐。

官方依据：[Nsight Systems 采集与分析](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)。

## 工具练习 C：NVTX 标注阶段

在 P03/P06 学会 NVTX，自己给初始化、warmup、measure、validate 四段加命名范围；在 P16 用 layer1/gemm、layer1/bias_relu、layer2/gemm 等阶段名，P20 加 chunk ID 和 stream ID，P22 加 capture/replay。

需要核对实际 NVTX 头文件和库组织；NVTX3 常用头文件接口，旧 NVTX 接口可能有不同链接要求，按安装版本接入，保持 CPU-only 可构建。先使用成对 push/pop 或已有 C++ 范围对象，确保提前退出时范围关闭，不提供完整实现让你抄写。

NVTX 范围默认标记 CPU 区间；CPU 结束范围不代表所提交 GPU 任务已经完成。将它关联到 GPU 工作看依赖，不能为让条带看起来对齐而增加同步改变程序性能。分析模式和性能模式可以通过编译/运行开关控制标注。

交付：一张带四阶段名称的时间线和一句话解释每阶段的 CPU/GPU 对应关系。API 来源与范围细节见 [NVTX 官方文档](https://nvidia.github.io/NVTX/)。

## 工具练习 D：Nsight Compute 看一个 Kernel

### 从少量 section 开始：P04

section 是一组相关指标，不等于单一硬件计数器。先查询安装版本支持项：

```bash
ncu --list-sets
ncu --list-sections
ncu --query-metrics
```

从 `basic` set 入手，先看 launch 配置、资源和占用率；需要深入时从已列出的 section 选择 MemoryWorkloadAnalysis、SchedulerStats、WarpStateStats、SourceCounters 等对应项。名字与 GPU/版本有关，缺失不能生搬硬套其他 GPU 的 metric ID，不要求一开始 `--set full`。

已有 P01 kernel 可作单 launch 的入门采集：

```bash
ncu --set basic --kernel-name regex:fma_kernel \
  --launch-skip 20 --launch-count 1 \
  -o reports/results/tooling/p01-fma-basic \
  ./build/cuda-sm89/bin/p01_bench_fma
```

这个示例选的是第一个配置下、20 次匹配的预热之后的一个 FMA launch，不会自动选 256 MiB 或 256 blocks。ncu 可能重放这个 launch；`launch-count=1` 不表示只收一轮硬件计数，也不表示应用剩余部分一定不运行。后续先固定配置再采集。

用 ncu GUI 打开 .ncu-rep，或按版本 CLI 导入显示；报告列出所选 kernel、调用序号、grid/block、shape、section、replay 与缓存/时钟控制设置。

### 需要解读的证据

| 阶段 | 重点观察 | 必须解释 |
|---|---|---|
| P04 | 寄存器、理论/实际占用率、活跃线程及分支相关信息 | 资源是否限制驻留？编译器是否保留所研究分支？高占用是否更快？ |
| P05/P07 | 全局访问效率、L1/L2/DRAM、共享内存 bank conflict | 哪一级流量减少？填充或 tile 是否改变冲突和资源？ |
| P09 | 原子相关执行与内存开销、活跃 warp | 冲突分布与耗时是否对应？不要给工具未提供的原子计数编数字 |
| P10/P11 | 同步、warp 调度、尾部有效线程 | 等待的原因和归约/scan 阶段在哪里？ |
| P14/P17 | 不均衡任务、L2 命中与实际 DRAM 流量 | 按算法访问量算的 GB/s 是否等于实测显存流量？ |
| P16/PA | GEMM 计算与访存利用率、实际指令路径 | 是否实际使用目标低精度/Tensor Core 路径？不是只看 dtype |

从 ncu Roofline（计算吞吐和访存带宽共同限制的上界模型）或自己按已声明 FLOP/字节画位置开始，分别说明算法算术强度和工具实测流量口径。不要只看某个 stall 百分比就认定根因；对照调度、访存和正常耗时来提出一个可检验的改动。

### 缓存和重放特别练习：P05/P06

P01 的 16 MiB D2D 超过标准显存带宽只是有效复制数据量高，不证明 DRAM 超越物理上限；拷贝引擎传输也不能随意当成可由 ncu 捕获的用户 kernel。用 FMA/转置在小大工作集上的 kernel 练习观察 L2 与 DRAM。

ncu 默认的缓存清理、时钟控制及 launch 串行化可能让结果与普通 benchmark 不同。先记录默认设置，再在支持且允许的情况下用相同 kernel 比较 `--cache-control all` 与 `none`；kernel replay 的 none 也不等于原应用稳态缓存。需要保留应用预热时研究 application replay，并保证整个程序可重复和确定性。不要自动修改服务器 GPU 时钟或共享环境配置。

分析工具下的耗时用于理解，最终性能结论仍来自不带 profiler 的 20/100 测量。记录“差异可能由工具条件引起”，不要求两者时间完全一致。

官方依据：[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)、[重放与测量条件](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 工具练习 E：编译资源和设备指令

P04 给实际 CUDA target 加资源报告选项，记录 64/128/256/512 threads/block 的寄存器和 shared memory。block 大小变了不一定使同一 kernel 的每线程寄存器数变化，必须读取编译证据。

P05 观察 tile=16/32 版本的寄存器、共享内存和 spill。spill 是寄存器放不下时部分值转移到 local memory 的访问；local memory 位于设备内存体系中，不是 shared memory。资源日志中的 stack/local 使用不能一概都称为 spill，结合 spill loads/stores 和指令判断。

先用实际已有的 ELF 程序练习：

```bash
cuobjdump --dump-resource-usage ./build/cuda-sm89/bin/p01_bench_fma \
  > reports/results/tooling/p01-resources.txt
cuobjdump --dump-sass ./build/cuda-sm89/bin/p01_bench_fma \
  > reports/results/tooling/p01-sass.txt
```

P04 找到条件判断与分支/谓词，P05 找 global/shared 载入和 barrier，PA 找实际矩阵计算指令。指令名随架构变化，先定位函数和源行，不要求背完整指令表。nvdisasm 的输入通常是 cubin，而不是直接传整个 host ELF；需要它时先通过实际工具选项提取 cubin，记录命令。PTX 并不总会保留在已构建文件中，不存在时不能以空输出作结论。

官方依据：[CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。

## 工具练习 F：交互调试（P03 必须掌握一次，工具不可用则设计演练）

制作独立的小调试输入，使用独立设备调试构建，在 cuda-gdb 中：

1. 在 kernel 首条计算语句设断点，运行小于一个 block 的输入。
2. 查看选中的 block/thread、n、全局下标，观察一个有效线程与一个越界线程的路径。
3. 单步跟踪一次载入、计算与写回，检查 out。
4. 保存命令记录与变量观察，退出调试后用 Release 重新验证正确性和性能。

按版本 `help` 学习 `info cuda threads` 等命令和线程选择方法；不要求调试器手工单步每个线程。在共享内存协作 kernel 中，随意单步与断点可能改变调度，先用最简单向量任务入门。

P01 CPU 初始化循环把字节数当 float 元素数导致的段错误，是主机错误。作为选做，在独立故障样例用主机 gdb backtrace 或 AddressSanitizer 定位；GPU memcheck 的无错误不能证明 CPU 内存访问安全。

官方依据：[CUDA-GDB 使用文档](https://docs.nvidia.com/cuda/cuda-gdb/index.html)。

## 各题的工具交付

各题 README 的“工具练习”指定最小交付，工具可用时属于练习验收，不是只写“用了 profiler”。完整 sweep 不必都 profile：选一个边界案例和一到两个有代表性的性能配置。P01 已完成的原作业保持完成；工具回顾可在 P02–P06 中补做，不追溯否定此前结果。

工具缺失时，记录命令、错误、版本/路径检查和预期观察；写出如何采集及如何判断的方案，硬件指标写未测。工具恢复可用后再补证据。不因故障而编造通过，也不让可完成的 CPU/正确性部分停下来。

工具报告保存到 `reports/results/pXX/tools/`，故障样例可放 `benchmarks/tooling/`，只在明确的独立目标中构建；默认构建不引入故障程序。总记录可以放 `reports/tooling.md`，每题报告链接对应证据。文件较大不提交 Git，下载时还要保留源码/路径映射和工具版本。

每次分析只写一个证据链：

```text
问题/预测 → 固定输入与配置 → 正常基线时间 → 采集命令和报告
→ 观察到的事实 → 尚未确认的原因 → 只改一个变量
→ 正确性复验 → 正常环境重新测速 → 接受或否定预测
```

完成此路线应能独立：定位一个内存错误、读一次 CPU/GPU 时间线、解读一个 kernel 的资源/访存信息、区分采集扰动与程序性能、用证据验证一次优化。
