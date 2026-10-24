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
