
#pragma once
// =============================================================================
// coalesced.cuh -- Kernel 2: coalesced global memory access (Part 2)
// =============================================================================
// Row-major, FP32:
// C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
// =============================================================================

#include "../common.cuh"

// Each thread computes one element of C.
// Consecutive threads are mapped to consecutive columns.
__global__ void sgemm_coalesced(
    int M, int N, int K, float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Consecutive threads access consecutive columns.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check.
    if (row >= M || col >= N) {
        return;
    }

    // Same matrix multiplication as the naive kernel.
    float acc = 0.0f;

    for (int k = 0; k < K; ++k) {
        acc += A[(size_t)row * K + k] *
               B[(size_t)k * N + col];
    }

    // C = alpha * A @ B + beta * C
    size_t idx = (size_t)row * N + col;

    if (beta == 0.0f) {
        C[idx] = alpha * acc;
    } else {
        C[idx] = alpha * acc + beta * C[idx];
    }
}

// CUDA kernel launcher.
bool launch_coalesced(
    int M, int N, int K, float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t s
) {
    // 32 consecutive columns per block.
    dim3 block(32, 8);

    // Cover the entire output matrix.
    dim3 grid(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y
    );

    // Launch kernel on the provided CUDA stream.
    sgemm_coalesced<<<grid, block, 0, s>>>(
        M, N, K, alpha, A, B, beta, C
    );

    return true;
}


