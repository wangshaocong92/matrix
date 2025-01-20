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
 * @brief 矩阵乘法的实现
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