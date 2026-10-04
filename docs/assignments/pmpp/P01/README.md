# P01：认识你的 GPU，并测量数据搬运和简单计算

## 你要学会什么

完成后，你应该能说清楚自己的 GPU 是什么，CPU 和 GPU 之间复制数据需要多久，以及一个简单的乘加 kernel 能达到多少实际吞吐。这里的 H2D 是 host（CPU 内存）到 device（GPU 显存），D2H 是反方向，D2D 是显存内部复制。

FMA 指 fused multiply-add，即 a*b+c。报告里的 FMA throughput 是这个程序实际测到的 GFLOP/s，不是 GPU 规格表上的理论峰值。

## 开始前

阅读 PMPP 第 1–2 章，先运行：

~~~bash
cmake --preset cuda-sm89
cmake --build --preset cuda-sm89 --parallel
./build/cuda-sm89/bin/device_inspect --json --require-gpu
~~~

GB10 主机把 cuda-sm89 换成 cuda-sm121。把输出保存为 reports/results/p01-device.json。

## 按步骤完成

1. 写 CPU reference：c[i] = a[i] * b[i] + c[i]。
2. 写 CUDA kernel，使用 grid-stride loop，让一个线程可以处理多个元素。
3. 为 H2D、D2H、D2D 各写一个 benchmark。每次先预热 20 次，再测 100 次。
4. 用 CUDA Event 计时，并在每次测量前后正确同步；不要用一个 CPU 计时器包住异步 API。
5. 测量 1 MiB、16 MiB、256 MiB 三个输入。记录中位数、最小值和有效带宽。
6. 测 FMA kernel，至少使用三个输入规模，按 2 FLOP/FMA 计算 GFLOP/s。
7. 在 4090D 和 GB10 分别运行；没有实测的一列写“未测”，不要填估算值。
8. 用随机输入把 GPU 结果与 CPU 结果逐元素比较，并设置清楚的误差容限。

## 需要修改的文件

- benchmarks/p01/fma.h、fma.cu：CPU reference、GPU kernel 与启动接口。
- benchmarks/p01/cuda_check.h：CUDA 错误检查。
- benchmarks/p01/test_fma.cpp：独立正确性测试。
- benchmarks/p01/bench_h2d.cpp、bench_d2h.cpp、bench_d2d.cpp、bench_fma.cpp：独立测速程序。
- benchmarks/CMakeLists.txt
- reports/p01.md

## 独立运行

P01 已由单一 main.cpp 拆为库、测试和四个 benchmark，使用现有目标：

```bash
./build/cuda-sm89/bin/p01_test_fma
./build/cuda-sm89/bin/p01_bench_h2d
./build/cuda-sm89/bin/p01_bench_d2h
./build/cuda-sm89/bin/p01_bench_d2d
./build/cuda-sm89/bin/p01_bench_fma
```

当前 4090D 部分已完成，GB10 未测。教学和实验遵循 [统一规则](../RULES.md)，实现由学习者完成，指导者按阶段检查。

## CUDA 工具练习

按 [工具练习路线](../TOOLING.md) 学习命令、编译方式和指标解释。先正确性，再整体时间线，最后按问题采集单 kernel；普通 benchmark 与工具运行分开。

1. 补充回顾，不改变已经完成状态：用 nsys 查看当前 FMA 程序中恢复 c 的 H2D、kernel 和同步间隔，分别辨认 CPU API 时间与 GPU 时间。
2. 用 memcheck 检查现有 p01_test_fma；用 ncu 只采一个明确 FMA 配置，对照工具下与正常 benchmark 的时间，记录缓存/时钟/replay 设置。不能用 ncu 的用户 kernel 计数器直接解释所有 cudaMemcpy D2D。

交付：在本题报告列工具版本、实际命令、配置、报告/日志路径、观察事实和下一步假设。工具可用时完成本节是练习验收的一部分；不可用则保存错误和采集方案，指标写未测，不阻塞其他可完成阶段。新工具练习不追溯否定 P01 已完成状态，P21/PA 仍为选做。


## 完成标准

- CPU 和 GPU 结果一致，空输入和小输入不会越界。
- 三种数据规模都有 H2D、D2H、D2D 数据。
- FMA 有 GFLOP/s 数据，并写明计算公式。
- 报告包含 GPU 型号、驱动/Toolkit、编译命令、warmup、iterations 和计时方法。
- 别人只看报告中的命令，就能重新运行实验。

## 常见问题

- 时间接近 0：通常是忘了同步或只测了 kernel launch。
- 每次结果差很多：增加预热和迭代次数，报告中位数。
- 把 GB/s 当成固定规格：它是本次程序、输入规模和计时方式得到的有效带宽。
- 只有一张卡：另一张卡写未测，不要用网上数字代替。
