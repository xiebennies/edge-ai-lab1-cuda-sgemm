
#pragma once
// =============================================================================
// smem.cuh -- Kernel 3: shared-memory tiling (Part 3a)
// =============================================================================
#include "../common.cuh"

template <int TILE>
__global__ void sgemm_smem(
    int M, int N, int K, float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
) {
    // Shared memory tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // Thread position inside the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output matrix position
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;

    // Iterate over K in TILE-sized chunks
    for (int t = 0; t < K; t += TILE) {

        // Load a tile of A into shared memory
        if (row < M && (t + tx) < K) {
            As[ty][tx] = A[(size_t)row * K + (t + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load a tile of B into shared memory
        if ((t + ty) < K && col < N) {
            Bs[ty][tx] = B[(size_t)(t + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Wait until all threads finish loading
        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        // Wait before overwriting shared memory
        __syncthreads();
    }

    // Write result back to global memory
    if (row < M && col < N) {
        size_t idx = (size_t)row * N + col;

        if (beta == 0.0f) {
            C[idx] = alpha * acc;
        } else {
            C[idx] = alpha * acc + beta * C[idx];
        }
    }
}

template <int TILE>
bool launch_smem(
    int M, int N, int K, float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t s
) {
    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    sgemm_smem<TILE><<<grid, block, 0, s>>>(
        M, N, K, alpha, A, B, beta, C
    );

    return true;
}

