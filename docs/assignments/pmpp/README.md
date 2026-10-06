# PMPP 作业总目录

P01–P23 与 PA 已全部布置。按主题阅读 PMPP，不绑定某一版章节号；部分工程题是为 Tiny Transformer 扩展。先读 [统一实验与教学规则](RULES.md)，再逐阶段做题。文档已经写好不代表代码目标已经存在。

当前 P01 的 4090D 和 GB10 实验已完成，GB10 于 2026-10-06 在 spark 的 kernellab 容器中补测；继续按 P02 的实际完成阶段推进。P21 和 PA 选做。先完成 P01–P06，再按项目计划交替学习 PMPP 与 Modern GPU。

| 题号 | 内容 | 前置 | 类型 |
|---|---|---|---|
| [P01](P01/README.md) | 设备、数据搬运和 FMA | 无 | 必做 |
| [P02](P02/README.md) | 一维向量与线程索引 | P01 | 必做 |
| [P03](P03/README.md) | 二维索引、均值滤波和朴素矩阵乘法 | P02 | 必做 |
| [P04](P04/README.md) | 分支、线程块大小与占用率 | P02、P03 | 必做 |
| [P05](P05/README.md) | 共享内存转置与分块矩阵乘法 | P03、P04 | 必做 |
| [P06](P06/README.md) | 第一张可复现性能对照表 | P02–P05 | 必做 |
| [P07](P07/README.md) | 多通道二维卷积 | P03、P05 | 必做 |
| [P08](P08/README.md) | 多时间步 stencil | P05、P07 | 必做 |
| [P09](P09/README.md) | 原子直方图与局部合并 | P04、P05 | 必做 |
| [P10](P10/README.md) | 归约与 RMSNorm | P05、P09 | 必做 |
| [P11](P11/README.md) | Exclusive scan 与稳定压缩 | P10 | 必做 |
| [P12](P12/README.md) | 两个有序数组的并行合并 | P11 | 必做 |
| [P13](P13/README.md) | 整数 Radix sort 与 Top-k | P09、P11、P12 | 必做 |
| [P14](P14/README.md) | CSR 稀疏矩阵乘向量 | P10、P12 | 必做 |
| [P15](P15/README.md) | Frontier BFS 图遍历 | P09、P11、P14 | 必做 |
| [P16](P16/README.md) | 两层 Tiny MLP 与逐元素融合 | P05、P10 | 必做 |
| [P17](P17/README.md) | Embedding gather 与重复索引 scatter | P09、P13、P16 | 必做 |
| [P18](P18/README.md) | 粒子二维空间分桶与邻域查询 | P09、P11、P17 | 必做 |
| [P19](P19/README.md) | 按输入条件选择 kernel 的 Dispatcher | P05、P06、P16 | 必做 |
| [P20](P20/README.md) | 多 Stream 分块流水线 | P02、P19 | 必做 |
| [P21](P21/README.md) | Dynamic Parallelism：设备启动子 Kernel（选做） | P15、P20 | 选做 |
| [P22](P22/README.md) | CUDA Graph 与 Workspace 复用 | P16、P19、P20 | 必做 |
| [P23](P23/README.md) | 一个算子的完整实验复盘 | P06，以及选择题的已完成实现 | 必做 |
| [PA](PA/README.md) | 数值精度、速度与误差（选做） | P05、P10，建议 M06 已完成 | 选做 |

## CUDA 工具学习路线

详见 [CUDA 调试与性能分析工具练习](TOOLING.md)，每题已有对应任务：

| 学习阶段 | 练习题 | 主要工具与产出 |
|---|---|---|
| 越界、未初始化和同步检查 | P02、P05、P08、P10、P11 | Compute Sanitizer 日志、独立故障样例的修复前后对照 |
| 整体时间线与阶段标注 | P03、P06、P16 | Nsight Systems + NVTX，辨认 CPU 提交、GPU 执行和等待 |
| 单 kernel 资源和瓶颈 | P04、P05、P07、P09、P14、P17 | Nsight Compute sections、资源日志和一次可验证优化 |
| 源码到指令与交互调试 | P03、P04、P05、PA | cuda-gdb、ptxas、cuobjdump/nvdisasm |
| 不规则、多阶段工作量 | P11–P15、P18、P19 | 阶段时间、负载与访问证据 |
| 并发与固定推理流程 | P20、P22 | stream/Graph 时间线与正常端到端复测 |
| 完整分析复盘 | P23 | 假设—证据—改动—正确性—性能复验 |

先定位瓶颈再选择工具，不要求对全部 sweep 收集 full 指标。远程 CLI 采集、本机 GUI 阅读；工具暂不可用则写明限制和补采方案。
