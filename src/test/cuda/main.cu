#include "matmul.cuh"
#include "helper.cuh"
#include <array>
// #include <cstdio>
// #include <iostream>
// #include <string>
// #include <sys/types.h>


int main()
{
  constexpr int M = 4, N = 4,K = 4;
  std::array<float, M * N> a, b, c;
  for(auto  i = 0;i <M * N; i++)
  {
    a[i] = 1.0f; 
    b[i] = 2.0f; 
  }
  c.fill(0);
  float *d_A, *d_B, *d_C;
  
  CUDA_CHECK(cudaMalloc(&d_A,sizeof(float) * M * N));
  CUDA_CHECK(cudaMemcpy(d_A, a.data(), sizeof(float) * M * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_B,sizeof(float) *M * N));
  CUDA_CHECK(cudaMemcpy(d_B, b.data(), sizeof(float) *M * N, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&d_C,sizeof(float) *M * N));
  dim3 block(32, 32);
  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  navive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
  
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}
  CUDA_CHECK(cudaMemcpy(c.data(), d_C, sizeof(float) *M * N, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  for(auto i = 0;i< M;i ++)
  {
    for(auto j = 0;j< N;j ++)
  {
    std::cout << std::to_string(c[i * N + j]) << " ";
  }
  std::cout << std::endl;
  }
  return 0;
}