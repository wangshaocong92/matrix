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
 * @brief 将被乘矩阵转置后再进行矩阵乘法，空间局部性原则
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