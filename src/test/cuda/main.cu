#include "helper.cuh"
#include "matmul.cuh"
#include <cstdint>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
class MutAnalogMatrix {
public:
    MutAnalogMatrix(int M, int N, int K)
        : M(M)
        , N(N)
        , K(K) {
        a.reserve(M * N);
        b.reserve(M * N);
        for (auto i = 0; i < M * N; i++) {
            a.push_back(1.0f + rand() % 10 / 10.0f);
            b.push_back(2.0f + rand() % 20 / 20.0f);
        }
        CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * K));
        CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K * N));
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * N));
    }

    ~MutAnalogMatrix() {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

public:
    float *d_A;
    float *d_B;
    float *d_C;

private:
    std::vector<float> a;
    std::vector<float> b;
    int M;
    int N;
    int K;
};

int main() {
    constexpr int M = 128, N = 128, K = 128;
    GpuTimer timer;
    int repeat_times = 50;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };
    std::cout << "navive_matmul \n";
    for (auto i = 0; i < 6; i++) {
        uint64_t m = M << i;
        uint64_t n = N << i;
        uint64_t k = K << i;
        int64_t flops = 2 * m * n * k;
        MutAnalogMatrix mat(m, n, k);
        timer.start();
        dim3 blockDim(32, 32);
        dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);
        int tmp_repeat_times = repeat_times;
        for (; tmp_repeat_times--;) {
            navive_matmul<<<gridDim, blockDim>>>(mat.d_A, mat.d_B, mat.d_C, m, n, k, 1.0f, 0.0f);
        }
        cudaDeviceSynchronize();
        timer.stop();
        auto &&elapsed_time = timer.elapsed_millis();
        std::cout << "mut size :" << m << " " << "Time: " << elapsed_time / repeat_times
                  << "ms GFLOPS:" << (flops * repeat_times * 1e-9) / (elapsed_time / 1000)
                  << std::endl;
    }

    std::cout << "sgemm_gmem_coalesce \n";
    for (auto i = 0; i < 6; i++) {
        uint64_t m = M << i;
        uint64_t n = N << i;
        uint64_t k = K << i;
        int64_t flops = 2 * m * n * k;
        MutAnalogMatrix mat(m, n, k);
        dim3 blockDim(32 * 32);
        dim3 gridDim((m + 32 - 1) / 32, (n + 32 - 1) / 32);
        int tmp_repeat_times = repeat_times;
        timer.start();

        for (; tmp_repeat_times--;) {
            sgemm_gmem_coalesce<32>
                <<<gridDim, blockDim>>>(m, n, k, 1.0f, mat.d_A, mat.d_B, 0.0f, mat.d_C);
        }
        cudaDeviceSynchronize();
        timer.stop();
        auto &&elapsed_time = timer.elapsed_millis();
        std::cout << "mut size :" << m << " " << "Time: " << elapsed_time / repeat_times
                  << "ms GFLOPS:" << (flops * repeat_times * 1e-9) / (elapsed_time / 1000)
                  << std::endl;
    }
    std::cout << "cublasGemmEx \n";
    for (auto i = 0; i < 6; i++) {
        uint64_t m = M << i;
        uint64_t n = N << i;
        uint64_t k = K << i;
        int64_t flops = 2 * m * n * k;
        MutAnalogMatrix mat(m, n, k);
        timer.start();
        int tmp_repeat_times = repeat_times;
        for (; tmp_repeat_times--;) {
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, mat.d_B, CUDA_R_32F, n,
                         mat.d_A, CUDA_R_32F, k, &beta, mat.d_C, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaDeviceSynchronize();
        timer.stop();
        auto &&elapsed_time = timer.elapsed_millis();
        std::cout << "mut size :" << m << " " << "Time: " << elapsed_time / repeat_times
                  << "ms GFLOPS:" << (flops * repeat_times * 1e-9) / (elapsed_time / 1000)
                  << std::endl;
    }
    return 0;
}