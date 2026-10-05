# PMPP 统一实验与教学规则

本路线来自项目学习计划的 P01–P23 和 PA，不是按某一版 PMPP 章节号逐一翻译。不同版本章节安排会变化，阅读按主题定位；题目中的 RMSNorm、MLP、Graph 等任务是为 Tiny Transformer 补充的工程练习。

## 分阶段教学

每题先明确输入输出和边界，写 CPU reference，再写最直白的 GPU baseline（基线，即用于比较的初始版本），先过正确性再改性能。学习者实现，指导者说明原理、API 和检查标准；默认不给完整答案，也不代写实现。一次只推进一个阶段、一次只改变一个实验变量。

遇到错误，先读编译或运行输出、定位原因并给局部提示。优化未加速也可完成，只要测试正确、实验口径一致、分析有证据。P21 和 PA 选做，不阻塞主线；GB10 当前不可用，所有题报告明确写“未测”。

## 共用术语

- kernel：在 GPU 上执行的函数；CPU reference：用于确认正确结果的 CPU 实现。
- thread/block/grid：线程、线程块和本次启动的全部线程块；warp：通常为一起执行的 32 个线程。
- SM：GPU 的流式多处理器，调度并执行线程块。
- tile：将矩阵或区域划成的小块；shared memory：一个 block 内共享的片上存储；register：线程私有的寄存器。
- shape/dtype：数据形状/元素类型；launch：提交一次 kernel 执行。
- H2D/D2H/D2D：CPU 到 GPU、GPU 到 CPU、同 GPU 显存内部的数据复制。
- FLOP/GFLOP/s：浮点运算次数/每秒十亿次浮点运算；bandwidth：单位时间处理的数据字节量。
- profiler：性能分析工具；baseline：用于后续对照的初始正确实现。

## 文件与构建目标

实现放 benchmarks/pXX/（PA 为 benchmarks/pa/），公共接口 .h 不含 CUDA 依赖，CPU reference 单独 .cpp，GPU 实现 .cu，测试和 benchmark 各有独立入口。每题有指定模块名，用模块名.h、模块名_cpu.cpp、模块名.cu。只有测试并确实要复用的实现才移入公共库。

通常将 CPU 与 GPU 代码建成库，独立程序链接它们；统计复用 gpulab::bench_stats。CUDA 目标链接 `gpulab::cuda_check`，包含 `gpulab/cuda_check.h` 使用 `CUDA_CHECK`；公共头文件位于 `include/gpulab/cuda_check.h`，不复制到每题，也不从其他题目目录包含。

题目约定 pXX_test 和 pXX_bench 作为未来目标名（PA 对应 pa_test/pa_bench）。P01/P02 保留已经布置的名称。只把实际完成的源文件显式加入 CMake，不用 GLOB 扫描空文件，不在当前工程提前创建所有目标。文档出现的未来目标命令须等对应 target 实现后执行。P06/P23 为报告整合题，复用原程序，不强行新建可执行程序。

CPU reference、纯 CPU 测试使用 cpu-debug，无 CUDA 依赖。GPU target 只在 KERNELLAB_ENABLE_CUDA 开启时构建。正确性程序接入 CTest 时，CPU 测试与 GPU 测试区分；GPU 测试受 KERNELLAB_ENABLE_GPU_TESTS 控制。可通过明确 --cpu-only/--gpu 模式拆分注册，或者使用两个测试程序，不能让 CPU-only 的通过被解释为 GPU 已测试。复用项目缺少 GPU 时退出码 77 的约定。

## 正确性

每题首先在纸上或 CPU 上完成一个手算小输入；覆盖空输入、单元素、非整齐长度、随机输入和题目指定边界。整数结果精确比较，float 逐元素比较并记录容限，初始一般采用 abs_error <= 1e-5 + 1e-5*abs(reference)，长归约/GEMM 按累积长度检查实际误差，不为了通过而盲目放宽。题目另定容限时以题目为准。

随机种子固定并记录。非有限数（NaN/Inf）是否支持由每题合同决定；要求传播时单独判断，不能靠普通差值比较。多数题先限定有限输入，P10 和 PA 专门覆盖数值异常。

检查每次返回 cudaError_t 的 CUDA API。kernel 启动后检查启动错误，获取结果前同步并检查执行错误。失败输出规模、版本、位置和实际/参考值，返回非零退出码；完成清理再退出，或者由现有错误宏终止进程。支持时运行 Compute Sanitizer（显存越界、竞态等检查工具），不可用写“未运行”，不得编造工具通过。竞态检查并不覆盖所有可能的全局数据竞态。

## 性能测量

默认每配置预热 20 次、测量 100 次，使用 CUDA Event 记录 GPU 工作区间。start、被测工作、stop 在同一 stream（有序任务队列）或有明确的跨流依赖；等待 stop 完成后取耗时，单位 ms。多阶段算法必须计入全部阶段；分配、初始化、传输是否纳入要按题目明确写出。

会修改输入或累积输出的算法每次恢复同一初值。计时外恢复也会影响缓存状态，报告写明复用和恢复方式。streams/Graph 等端到端题额外使用 CPU 单调时钟从提交前计到全部 GPU 工作完成，不能拿 CPU 计时包住未同步的异步提交。

复用当前 nearest-rank 中位统计：排序后取 ceil(0.5*N) 位，100 个样本取下标 49；不称为中间两值平均。报告 min 与 median，只有讨论尾延迟时再加 p95/最大值。小输入接近计时分辨率时标注，必要时测整批多次执行再除次数，同时保持原始和批量测法分开记录。

MiB=2^20 bytes，GB/s 采用 10^9 bytes，GFLOP/s 采用 10^9 浮点运算。公式的毫秒转换写成除以 ms*10^6。每题列算法约定字节数和 FLOP；缓存命中、重复访问、元数据可能使约定量不同于实际显存流量，不能直接当硬件计数值。

## 环境与原始记录

4090D 用 cuda-sm89；GB10 可用后用 cuda-sm121，不支持时不强行切换其他架构。记录 GPU 实际名称、计算能力、实际显存、驱动、Toolkit、编译器、CPU/OS、CMake preset、启动配置、dtype、shape、warmup、iterations 和计时方法。P01 本机实际显存查询与标准规格不同，后续沿用设备实际信息，不默认填宣传规格。

报告放 reports/pXX.md（PA 为 reports/pa.md），原始输出放 reports/results/pXX/。可基于 ../template/REPORT.md，描述预测、结果、失败尝试和下一步。日志管道使用 pipefail，源码 Git 版本或 SHA-256 校验值一并保存。仓库约定 docs/ 本地维护不提交，reports/results/ 不提交； authored reports 可按项目流程提交。

每个题目具体路径和目标优先于本通用模式；没有实测的一列写“未测”，工具不支持写“当前环境不支持”。

## 工具练习验收

新增 [CUDA 调试与性能分析路线](TOOLING.md)，各题 README 指定工具任务，按 memcheck→nsys/NVTX→ncu→编译资源/设备指令逐步推进，cuda-gdb 完成一次小输入交互调试。主性能用不带 profiler 的 20/100 测量；工具用于定位和解释，不直接替代主结果。

工具可用时，每题要求的日志、报告与证据是验收的一部分；缺失或无权限时提交版本检查、错误日志和采集方案，指标注明未测。P01 已完成状态不追溯撤销，P21/PA 保持选做。分析与 Debug 配置分开，不自动安装工具或改变共享服务器的驱动权限/时钟。
