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
