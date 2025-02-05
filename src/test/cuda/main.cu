#include "helper.cuh"
#include "matmul.cuh"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define FLOAT_EQUAL(a, b) (a - b < 1e-1 && a - b > -1e-1)
class MutAnalogMatrix {
public:
    MutAnalogMatrix(int M, int N)
        : M(M)
        , N(N) {
        c.resize(M * N);
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * N));
    };
    MutAnalogMatrix(int M, int N, int K)
        : M(M)
        , N(N)
        , K(K)
        , aandb(true) {
        a.reserve(M * K);
        b.reserve(K * N);
        c.resize(M * N);
        for (auto i = 0; i < M * K; i++) {
            a.push_back(1.0f + rand() % 10 / 10.0f);
            // a.push_back(1.0f);
        }
        for (auto i = 0; i < N * K; i++) {
            b.push_back(2.0f + rand() % 20 / 20.0f);
            // b.push_back(2.0f);
        }
        CUDA_CHECK(cudaMalloc(&d_A, sizeof(float) * M * K));
        CUDA_CHECK(cudaMalloc(&d_B, sizeof(float) * K * N));
        CUDA_CHECK(cudaMemcpy(d_A, a.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, b.data(), sizeof(float) * N * K, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_C, sizeof(float) * M * N));
        CUDA_CHECK(cudaMemset(d_C, 0, sizeof(float) * M * N));
    }

    bool operator==(const MutAnalogMatrix &other) const {
        if (M != other.M || N != other.N) {
            return false;
        }
        bool res = true;
        for (auto i = 0; i < M * N; i++) {
            if (!FLOAT_EQUAL(c[ i ], other.c[ i ])) {
                std::cout << "c[" << i << "]:" << c[ i ] << " other.c[" << i << "]:" << other.c[ i ]
                          << std::endl;
                res = res ? false : res;
                return false;
            }
        }
        return res;
    }
    void SaveResult() {
        CUDA_CHECK(cudaMemcpy(c.data(), d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    }

    ~MutAnalogMatrix() {
        if (aandb) {
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
        }

        CUDA_CHECK(cudaFree(d_C));
    }

public:
    float *d_A;
    float *d_B;
    float *d_C;

private:
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
    bool aandb = false;
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
#if 0
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

    std::cout << "sgemm_1d_blocktiling \n";
    for (auto i = 0; i < 6; i++) {
        uint64_t m = M << i;
        uint64_t n = N << i;
        uint64_t k = K << i;
        int64_t flops = 2 * m * n * k;
        const uint TM = 8;
        MutAnalogMatrix mat(m, n, k);
        dim3 blockDim(32 * 32 / TM);
        dim3 gridDim((n + 32 - 1) / 32, (m + 32 - 1) / 32 / TM);
        int tmp_repeat_times = repeat_times;
        timer.start();

        for (; tmp_repeat_times--;) {
            sgemm_1d_blocktiling<TM>
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

#endif

    //// 正确与否判断
    {
        ///// 1024 矩阵
        for (auto i = 0; i < 6; i++) {
            uint64_t m = M << i;
            uint64_t n = N << i;
            uint64_t k = K << i;
            // uint64_t m = 64;
            // uint64_t n = 64;
            // uint64_t k = 64;

            MutAnalogMatrix mata(m, n, k);

            {
                cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, mata.d_B,
                             CUDA_R_32F, n, mata.d_A, CUDA_R_32F, k, &beta, mata.d_C, CUDA_R_32F, n,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
                cudaDeviceSynchronize();
                mata.SaveResult();
            }

            {
                MutAnalogMatrix matb(m, n);
                const uint TM = 8;
                dim3 blockDim(32 * 32 / TM);
                dim3 gridDim((n + 32 - 1) / 32, (m + 32 - 1) / 32 / TM);
                sgemm_1d_blocktiling<TM>
                    <<<gridDim, blockDim>>>(m, n, k, 1.0f, mata.d_A, mata.d_B, 0.0f, matb.d_C);
                cudaDeviceSynchronize();
                matb.SaveResult();
                if (mata == matb)
                    std::cout << m << " : cublasGemmEx and sgemm_1d_blocktiling is equal"
                              << std::endl;
                else
                    std::cout << m << " : cublasGemmEx and sgemm_1d_blocktiling is not equal"
                              << std::endl;
            }
            {
                MutAnalogMatrix matc(m, n);
                dim3 blockDim(32 * 32);
                dim3 gridDim((n + 32 - 1) / 32, (m + 32 - 1) / 32);
                sgemm_share<<<gridDim, blockDim>>>(m, n, k, 1.0f, mata.d_A, mata.d_B, 0.0f,
                                                   matc.d_C);
                cudaDeviceSynchronize();
                matc.SaveResult();
                if (mata == matc)
                    std::cout << m << " : cublasGemmEx and sgemm_share is equal" << std::endl;
                else
                    std::cout << m << " : cublasGemmEx and sgemm_share is not equal" << std::endl;
            }
            {

                MutAnalogMatrix matd(m, n);
                dim3 blockDim(32 * 32);
                dim3 gridDim((m + 32 - 1) / 32, (n + 32 - 1) / 32);
                sgemm_gmem_coalesce<32>
                    <<<gridDim, blockDim>>>(m, n, k, 1.0f, mata.d_A, mata.d_B, 0.0f, matd.d_C);
                cudaDeviceSynchronize();
                matd.SaveResult();
                if (mata == matd)
                    std::cout << m << " : cublasGemmEx and sgemm_gmem_coalesce is equal"
                              << std::endl;
                else
                    std::cout << m << " : cublasGemmEx and sgemm_gmem_coalesce is not equal"
                              << std::endl;
            }
        }
    }

    return 0;
}