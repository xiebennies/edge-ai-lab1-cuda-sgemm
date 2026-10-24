# Edge AI Lab 1: CUDA SGEMM Optimization

FP32 SGEMM (`C = alpha * A @ B + beta * C`) implemented step by step in CUDA:
naive -> coalesced -> shared memory -> 2D register tiling -> custom optimizations.

## Results (M=N=K=4096)
| Kernel | Time | Speedup vs previous |
|---|---|---|
| reg2d_128x128x8_8x8 | 29.68 ms | baseline |
| custom_v2 (vec4 loads) | 26.28 ms | 1.13x |
| custom_v3 (unrolling) | 24.55 ms | 1.07x |

All kernels pass correctness on 15 shapes x 2 betas (including edge cases such as 1x1x1 and 1023x257x511) and compute-sanitizer runs.

## Layout
- `src/` kernels and benchmark harness
- `scripts/` correctness check, sweep, profiling helpers
- `results/` correctness and benchmark logs
- `benchmark_part1.txt` Part 1 benchmark log

## Build & Run
Requires CUDA >= 11.6 and cuBLAS.

    make
    ./sgemm ...

## Hardware
NVIDIA GeForce RTX 3060 Laptop GPU (sm_86, 30 SMs), WSL2 Ubuntu 22.04

## Profiling Notes (Nsight Compute, 2048^3)

| Kernel | Duration (ms) | vs previous | Compute % | Memory % | DRAM % | L1/TEX % | Achieved Occ. % |
|---|---|---|---|---|---|---|---|
| naive | 167.03 | - | 11.96 | 98.65 | 2.42 | 99.01 | 65.6 |
| coalesced | 20.48 | 8.16x | 99.64 | 99.64 | 73.85 | 99.78 | 99.5 |
| reg2d 64x64x8_4x4 | 4.46 | 4.59x | 59.07 | 74.12 | 12.10 | 74.31 | 81.3 |
| reg2d 128x128x8_8x8 | 4.52 | 0.99x | 51.33 | 59.35 | 12.57 | 63.62 | 31.6 |
| custom_v2 | 3.38 | 1.34x | 47.84 | 57.45 | 12.41 | 62.76 | 31.7 |

- **naive**: L1/TEX is saturated (99%) while DRAM is only 2.4% and compute 12%, so the bottleneck is the load path, consistent with uncoalesced global accesses.
- **coalesced**: 8.2x faster, but with no data reuse DRAM throughput reaches 74%. (Compute throughput is the max over pipelines, so 99.6% does not mean the FP32 units are saturated.)
- **reg2d**: shared-memory + register tiling cuts DRAM throughput to ~12%. The 128x128 tile reuses more data per thread, but its register usage limits occupancy to 2 blocks/SM (33% vs 83% for 64x64), so the two effects roughly cancel at this size (4.52 vs 4.46 ms, within run-to-run noise).
- **custom_v2**: same grid and occupancy as the 128x128 kernel but 25% faster with lower reported utilization, consistent with vec4 loads reducing the number of load instructions.
- naive -> custom_v2: ~49x at 2048^3.

Profiled with `--clock-control none` (clocks not locked), one run per kernel, so small differences are within noise. Durations under ncu differ slightly from the benchmark harness timings above. Raw output: `profiling_summary.txt`.
