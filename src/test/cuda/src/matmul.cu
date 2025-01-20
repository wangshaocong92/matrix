#include "matmul.cuh"
#include <iostream>

__global__ void navive_matmul(const float *A, const float *B, float *C, int M, int N, int K,
                              float alpha, float beta, cudaStream_t stream_id) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    // if statement is necessary to make things work under tile quantization
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            // printf("%d %f, %d %f\n", x * K + i ,A[ x * K + i ],i * N + y,B[ i * N + y ]);
            tmp += A[ x * K + i ] * B[ i * N + y ]; /// 行列相加
        }
        // C = α*(A@B)+β*C
        C[ x * N + y ] = alpha * tmp + beta * C[ x * N + y ];
    }
}