# Benchmark configurations

Reserved directory for the P01 workload configuration. `gpulab_bench` currently exposes only
`--list` and `--help`; it does not provide timing or real workloads yet.

Future configurations must explicitly include the workload, shape, dtype, seed, warmup, and
iteration count. Results must also record the machine and toolchain information.
