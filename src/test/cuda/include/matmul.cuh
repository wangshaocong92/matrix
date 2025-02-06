/**
 * @file matmul.h
 * @author wangshaocong (1635573386@qq.com)
 * @brief 基于cuda的多版本矩阵乘法实现
 * @version 0.1
 * @date 2025-01-20
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <sys/types.h>

/**
 * @brief 矩阵转置的实现
 *
 * @param out 转置后的矩阵
 * @param in 原矩阵
 * @param width 矩阵的宽度
 * @param height 矩阵的高度
 */

__global__ void transpose(float *out, const float *in, int width, int height);

/**
 * @brief 传统矩阵乘法的实现
 *
 * @param A 矩阵A
 * @param B 矩阵B
 * @param C 结果矩阵C
 * @param M 矩阵A的行数
 * @param N 矩阵A的列数
 * @param K 矩阵B的列数
 * @param alpha 系数
 * @param beta 系数
 * @param stream_id cuda流
 */
__global__ void navive_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                              float alpha, float beta, cudaStream_t stream_id = nullptr);
/**
 * @brief 将被乘矩阵转置后再进行矩阵乘法，空间局部性原则，看起来并没有什么卵用
 *
 * @param A 矩阵A
 * @param B 矩阵B
 * @param C 结果矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数，B的行数
 * @param alpha alpha
 * @param beta beta
 * @param stream_id cuda流
 * @return __global__
 */
__global__ void transpose_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                                 float alpha, float beta, cudaStream_t stream_id = nullptr);

/**
 * @brief
 *
 * @tparam BLOCKSIZE
 * @param M
 * @param N
 * @param K
 * @param alpha
 * @param A
 * @param B
 * @param beta
 * @param C
 * @return __global__
 */
template <const uint BLOCKSIZE = 32>
__global__ void sgemm_gmem_coalesce(int M, int N, int K, float alpha, const float *A,
                                    const float *B, float beta, float *C) {
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE); //// 获取当前warp的行号
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); //// 获取当前thread的列号

    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (
            int i = 0; i < K;
            ++i) { /// 同一warp
                   /// 的线程会处理结果的同一行的数据，他们乘矩阵的同一行，即此行原本需要从global加载
                   /// BLOCKSIZE
                   // 次的当前 仅需要加载一次。单次warp的执行会顺序加载被乘矩阵的连续BLOCKSIZE内存
            tmp += A[ cRow * K + i ] * B[ i * N + cCol ]; // warp 内的线程会同步执行同一条指令
        }
        C[ cRow * N + cCol ] = alpha * tmp + beta * C[ cRow * N + cCol ];
    }
}

template <const uint TM = 32, const uint BM = 32, const uint BN = 32, const uint BK = 32>
__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha, const float *A,
                                     const float *B, float beta, float *C) {

    __shared__ float As[ BM * BK ];
    __shared__ float Bs[ BK * BN ];

    /// thread 与 C 映射 单block的线程数为 BM * BN / TM,单线程处理C中block的竖着的TM个元素
    const int cRow = blockIdx.y * BM; /// 一个block对应C的行号
    const int cCol = blockIdx.x * BN; /// 一个block对应C的列号

    const int c_threadCol = threadIdx.x % BN; /// 一个block中的线程对应C的列号
    const int c_threadRow = threadIdx.x / BN; /// 一个block中的线程对应C的行号，单TM算作一行

    //// a 和 b block的起始位置
    const int aRow = blockIdx.y * BM; /// 一个block对应a的行号
    const int bCol = blockIdx.x * BN; /// 一个block对应b的列号

    float threadResults[ TM ]; /// 单个线程会计算TM个结果，将他们临时存储最后在写入C
    ///__shared__ float Cs[ BM * BK ];

    for (auto i = 0; i < TM; i++) {
        threadResults[ i ] = 0.0;
    }

    for (auto i = 0; i < K; i += BK) {
        /// 先将A和B的相应内存快加载到共享内存中
        /// A 单线程复制 TM长度的一列， 我们默认 BK <= BN & BK <= BM
        for (auto j = 0; j < TM; j++) /// 在同一个warp执行
        {
            auto s_row = c_threadRow * TM + j;
            if (c_threadCol < BK && aRow + s_row < M && i + c_threadCol < K && s_row < BM) {
                As[ s_row * BK + c_threadCol ] = A[ (aRow + s_row) * K + i + c_threadCol ];
            }
            if (bCol + c_threadCol < N && i + s_row < K && s_row < BK) {
                Bs[ s_row * BN + c_threadCol ] = B[ (i + s_row) * N + bCol + c_threadCol ];
            }
        }
        __syncthreads();

        for (auto k = 0; k < TM; ++k) {
            float tmp = 0.0;
            auto s_row = c_threadRow * TM + k;
            for (auto j = 0; j < BK; ++j) {
                tmp += As[ s_row * BK + j ] * Bs[ j * BN + c_threadCol ];
            }
            threadResults[ k ] += tmp;
        }
        /// 同一个block 不仅仅只有一个warp，所以需要同步
        __syncthreads();
    }

    for (auto i = 0; i < TM; ++i) {
        if (cRow + c_threadRow * TM + i < M && cCol + c_threadCol < N) {
            C[ (cRow + c_threadRow * TM + i) * N + cCol + c_threadCol ] =
                alpha * threadResults[ i ] +
                beta * C[ (cRow + c_threadRow * TM + i) * N + cCol + c_threadCol ];
        }
    }
}

template <const uint BM = 32, const uint BN = 32, const uint BK = 32>
__global__ void sgemm_share(int M, int N, int K, float alpha, const float *A, const float *B,
                            float beta, float *C) {

    __shared__ float As[ BM * BK ];
    __shared__ float Bs[ BK * BN ];

    /// thread 与 C 映射 单block的线程数为 BM * BN / TM,单线程处理C中block的竖着的TM个元素
    const int cRow = blockIdx.y * BM; /// 一个block对应C的行号
    const int cCol = blockIdx.x * BN; /// 一个block对应C的列号

    const int c_threadCol = threadIdx.x % BN; /// 一个block中的线程对应C的列号
    const int c_threadRow = threadIdx.x / BN; /// 一个block中的线程对应C的行号，单TM算作一行

    //// a 和 b block的起始位置
    const int aRow = blockIdx.y * BM; /// 一个block对应a的行号
    const int bCol = blockIdx.x * BN; /// 一个block对应b的列号
    float tmp = 0.0;
    if (cCol + c_threadCol < N && cRow + c_threadRow < M) {
        for (auto i = 0; i < K; i += BK) {
            /// 先将A和B的相应内存快加载到共享内存中
            auto s_row = c_threadRow;
            if (c_threadCol < BK && aRow + s_row < M && i + c_threadCol < K && s_row < BM) {
                As[ s_row * BK + c_threadCol ] = A[ (aRow + s_row) * K + i + c_threadCol ];
            }
            if (bCol + c_threadCol < N && i + s_row < K && s_row < BK) {
                Bs[ s_row * BN + c_threadCol ] = B[ (i + s_row) * N + bCol + c_threadCol ];
            }
            __syncthreads();
            for (auto j = 0; j < BK; ++j) {
                tmp += As[ s_row * BK + j ] * Bs[ j * BN + c_threadCol ];
            }
            /// 同一个block 不仅仅只有一个warp，所以需要同步
            __syncthreads();
        }
        if (cRow + c_threadRow < M && cCol + c_threadCol < N) {
            C[ (cRow + c_threadRow) * N + cCol + c_threadCol ] =
                alpha * tmp + beta * C[ (cRow + c_threadRow) * N + cCol + c_threadCol ];
        }
    }
}

/**
 * @brief 2d blocktiling 矩阵乘法 block 线程数要和 BK * BM 和 BK * BN 一致。
 *   我们默认为 BM 和 BN 一致。而线程数量为 BN * BM / (TM * TN)。
 *   即 BN * BM / (TM * TN) == BK * BM 即 BK = BM / (TM * TN)
 *
 * @tparam TM
 * @tparam TN
 * @tparam BM
 * @tparam BN
 * @tparam BK
 * @param M
 * @param N
 * @param K
 * @param alpha
 * @param A
 * @param B
 * @param beta
 * @param C
 * @return __global__
 */
template <const uint TM, const uint TN, const uint BM, const uint BN, const uint BK>
__global__ __launch_bounds__(BM *BN / (TM * TN),
                             1) void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                                          const float *A, const float *B,
                                                          float beta, float *C) {
    __shared__ float As[ BM * BK ];
    __shared__ float Bs[ BK * BN ];

    // printf("thredIdx.x = %d\n", threadIdx.x);

    // assert(BK == BM / (TM * TN));
    // 我们不强制单个线程处理单个内存，这样会降低灵活性，我们可以强制一个
    assert(BM == BN);

    const int cRow = blockIdx.y * BM; /// 一个block对应C的行号
    const int cCol = blockIdx.x * BN; /// 一个block对应C的列号

    /// 内存copy和计算完全可以是不同的映射关系
    //// a 中单个线程加载的列个数
    auto a_signal_thread_load_row = BK * TM * TN / BM;
    auto amCol = threadIdx.x % BK;
    auto amRow = threadIdx.x / BK * a_signal_thread_load_row;

    //// b 中单个线程加载的列个数
    auto b_signal_thread_load_row = BK * TM * TN / BN;
    auto bmCol = threadIdx.x % BN;
    auto bmRow = threadIdx.x / BN * b_signal_thread_load_row;

    //// 计算映射
    const int c_sthreadCol = threadIdx.x % (BN / TN);
    const int c_sthreadRow = threadIdx.x / (BN / TN);

    float threadResults[ TM * TN ]{0.0}; /// 单个线程会计算TM个结果，将他们临时存储最后在写入C
    for (int i = 0; i < K; i += BK) {

        for (auto j = 0; j < a_signal_thread_load_row; j++) {
            As[ (amRow + j) * BK + amCol ] = A[ (cRow + amRow + j) * K + i + amCol ];
        }
        for (auto j = 0; j < b_signal_thread_load_row; j++) {
            Bs[ (bmRow + j) * BN + bmCol ] = B[ (i + bmRow + j) * N + cCol + bmCol ];
        }
        __syncthreads();

        for (int m = 0; m < BK; ++m) {
            float a[ TM ];
            float b[ TN ];
            for (int j = 0; j < TM; ++j) {
                a[ j ] = As[ (c_sthreadRow * TM + j) * BK + m ];
            }
            for (int j = 0; j < TN; ++j) {
                b[ j ] = Bs[ m * BN + c_sthreadCol * TN + j ];
            }
            for (int j = 0; j < TM; ++j) {
                for (int k = 0; k < TN; ++k) {
                    threadResults[ j * TN + k ] += a[ j ] * b[ k ];
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < TM; j++) {
        for (int k = 0; k < TN; k++) {
            auto col = (c_sthreadCol * TN) + k;
            auto row = (c_sthreadRow * TM) + j;
            if (cRow + row < M && cCol + col < N) {
                C[ (cRow + row) * N + cCol + col ] =
                    alpha * threadResults[ j * TN + k ] + beta * C[ (cRow + row) * N + cCol + col ];
            }
        }
    }
}

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta,
                       float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint totalResultsBlocktile = BM * BN;
    // A thread is responsible for calculating TM*TN elements in the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
    assert(numThreadsBlocktile == blockDim.x);

    // BN/TN are the number of threads to span a column
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // allocate space for the current blocktile in smem
    __shared__ float As[ BM * BK ];
    __shared__ float Bs[ BK * BN ];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // calculating the indices that this thread will load into SMEM
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    // calculates the number of rows of As that are being loaded in a single step
    // by a single block
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    // for both As and Bs we want each load to span the full column-width, for
    // better GMEM coalescing (as opposed to spanning full row-width and iterating
    // across columns)
    const uint strideB = numThreadsBlocktile / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[ TM * TN ] = {0.0};
    // register caches for As and Bs
    float regM[ TM ] = {0.0};
    float regN[ TN ] = {0.0};

    // outer-most loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[ (innerRowA + loadOffset) * BK + innerColA ] =
                A[ (innerRowA + loadOffset) * K + innerColA ];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[ (innerRowB + loadOffset) * BN + innerColB ] =
                B[ (innerRowB + loadOffset) * N + innerColB ];
        }
        __syncthreads();

        // advance blocktile
        A += BK;     // move BK columns to right
        B += BK * N; // move BK rows down

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // block into registers
            //// 获取Bm的Tm的列的值 即 Tm * 1
            for (uint i = 0; i < TM; ++i) {
                regM[ i ] = As[ (threadRow * TM + i) * BK + dotIdx ];
            }
            //// 获取Bn的Tn的行的值 1 * Tm
            for (uint i = 0; i < TN; ++i) {
                regN[ i ] = Bs[ dotIdx * BN + threadCol * TN + i ];
            }

            /// 相乘即是Tm * Tn的矩阵
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[ resIdxM * TN + resIdxN ] += regM[ resIdxM ] * regN[ resIdxN ];
                }
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[ (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN ] =
                alpha * threadResults[ resIdxM * TN + resIdxN ] +
                beta * C[ (threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN ];
        }
    }
}