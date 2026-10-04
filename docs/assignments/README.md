# 作业入口（白话版）

完整路线在 [学习计划](../LEARNING_PROJECT_PLAN.md)。每题都应该回答四个问题：我要学什么？先做哪一步？改哪些文件？什么结果算完成？

新增或改写题目时，请先阅读 [文档写作规范](../DOCUMENTATION_STYLE_GUIDE.md)，保持术语解释、步骤和验收标准的一致性。

- [pmpp/](pmpp/README.md)：P01–P23 和 PA 均有独立说明；P21、PA 选做。从 P01 开始，逐阶段推进。
- modern_gpu/：M01–M15，把 GPU 硬件知识连接到机器学习算子。
- b200_optional/：B01–B08，只有拿到 B200 后才做实机验证，不阻塞主线。
- template/REPORT.md：每题报告模板。里面的“性能账本”就是一张普通性能表。

建议顺序：先完成 P01–P06，再交替做 PMPP 和 Modern GPU，最后做 T0–T6。公共库只放已经测试、确实需要复用的实现；先在作业目录中完成实验。

注意：docs/ 按仓库约定只在本地维护，不提交到 Git。新 clone 后需要自行复制这部分学习文档。

CUDA 性能与调试工具随各题练习，入口为 [工具路线](pmpp/TOOLING.md)：Compute Sanitizer、Nsight Systems/NVTX、Nsight Compute、编译资源与 cuda-gdb。先做正确性，再用证据定位性能问题。
