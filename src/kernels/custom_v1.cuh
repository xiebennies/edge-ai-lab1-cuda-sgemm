
#pragma once
// =============================================================================
// Part 4 - V1: Shared Memory Padding
// Baseline: reg2d_128x128x8_8x8
//
// Optimization: pad the shared-memory row stride by one float.
// =============================================================================

#include "../common.cuh"

__global__ void __launch_bounds__(256)
sgemm_custom_v1(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int NT = 256;

    // V1 optimization: padded shared memory.
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];

    int tid = threadIdx.x;

    int tr = tid / (BN / TN);
    int tc = tid % (BN / TN);

    float acc[TM][TN] = {};
    float ra[TM];
    float rb[TN];

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // Load A tile into transposed shared memory.
        for (int idx = tid; idx < BM * BK; idx += NT)
        {
            int m = idx / BK;
            int k = idx % BK;

            As[k][m] =
                A[(size_t)(blockIdx.y * BM + m) * K
                  + k0 + k];
        }

        // Load B tile into shared memory.
        for (int idx = tid; idx < BK * BN; idx += NT)
        {
            int k = idx / BN;
            int n = idx % BN;

            Bs[k][n] =
                B[(size_t)(k0 + k) * N
                  + blockIdx.x * BN + n];
        }

        __syncthreads();

        // Same register-blocked outer product as Part 3b.
        for (int k = 0; k < BK; ++k)
        {
            for (int i = 0; i < TM; ++i)
                ra[i] = As[k][tr * TM + i];

            for (int j = 0; j < TN; ++j)
                rb[j] = Bs[k][tc * TN + j];

            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j)
                    acc[i][j] += ra[i] * rb[j];
        }

        __syncthreads();
    }

    // Write results to global memory.
    for (int i = 0; i < TM; ++i)
    {
        int row = blockIdx.y * BM + tr * TM + i;

        for (int j = 0; j < TN; ++j)
        {
            int col = blockIdx.x * BN + tc * TN + j;

            size_t idx = (size_t)row * N + col;

            if (beta == 0.0f)
                C[idx] = alpha * acc[i][j];
            else
                C[idx] = alpha * acc[i][j] + beta * C[idx];
        }
    }
}

// Part 4 - V1 launcher
inline bool launch_custom_v1(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C,
    cudaStream_t s)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;

    // Only accept supported matrix shapes.
    if (M % BM || N % BN || K % BK)
        return false;

    dim3 grid(N / BN, M / BM);

    sgemm_custom_v1<<<grid, 256, 0, s>>>(
        M, N, K, alpha, A, B, beta, C);

    return true;
}
