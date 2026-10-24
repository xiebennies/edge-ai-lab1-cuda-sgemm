
#pragma once
// =============================================================================
// Part 4 - V2: Vectorized Global Memory Loading (float4)
// Baseline: reg2d_128x128x8_8x8
// =============================================================================

#include "../common.cuh"

__global__ void __launch_bounds__(256)
sgemm_custom_v2(
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

    // Keep original shared-memory layout.
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;

    const int tr = tid / (BN / TN);
    const int tc = tid % (BN / TN);

    float acc[TM][TN] = {};
    float ra[TM];
    float rb[TN];

    for (int k0 = 0; k0 < K; k0 += BK)
    {
        // ---------------------------------------------------------
        // V2: Vectorized loading of matrix A.
        // Four consecutive K elements per float4.
        // ---------------------------------------------------------
        for (int idx = tid; idx < BM * (BK / 4); idx += NT)
        {
            int m = idx / (BK / 4);
            int k = (idx % (BK / 4)) * 4;

            size_t a_idx =
                (size_t)(blockIdx.y * BM + m) * K
                + k0 + k;

            float4 v =
                *reinterpret_cast<const float4*>(A + a_idx);

            As[k + 0][m] = v.x;
            As[k + 1][m] = v.y;
            As[k + 2][m] = v.z;
            As[k + 3][m] = v.w;
        }

        // ---------------------------------------------------------
        // V2: Vectorized loading of matrix B.
        // Four consecutive N elements per float4.
        // ---------------------------------------------------------
        for (int idx = tid; idx < BK * (BN / 4); idx += NT)
        {
            int k = idx / (BN / 4);
            int n = (idx % (BN / 4)) * 4;

            size_t b_idx =
                (size_t)(k0 + k) * N
                + blockIdx.x * BN + n;

            float4 v =
                *reinterpret_cast<const float4*>(B + b_idx);

            Bs[k][n + 0] = v.x;
            Bs[k][n + 1] = v.y;
            Bs[k][n + 2] = v.z;
            Bs[k][n + 3] = v.w;
        }

        __syncthreads();

        // Same register-blocked outer product.
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

    // Write TM x TN results back to global memory.
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

// Launcher
inline bool launch_custom_v2(
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

    // The supported dimensions ensure float4 row alignment
    // for the harness's CUDA-allocated matrices.
    if (M % BM || N % BN || K % BK)
        return false;

    dim3 grid(N / BN, M / BM);

    sgemm_custom_v2<<<grid, 256, 0, s>>>(
        M, N, K, alpha, A, B, beta, C);

    return true;
}
