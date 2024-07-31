#include <cstdio>
#include <cstdlib>
#include "sum_matrix.h"
#include <stdio.h>
#include <cuda_runtime.h>

void initData(float *f, int size, float value) {
  for (int i = 0; i < size; i++)
    *(f + i) = value;
}

void check_data(float *a, int n) {
  for (int i = 0; i < n; i++)
    printf("Current :%.3f\n", *(a + i));
}
int main() {
  int dev = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  int nx = 1 << 5;
  int ny = 1 << 5;

  int nxy = nx * ny;
  int nBytes = nxy * sizeof(float);

  float *h_a, *h_b, *hostRef, *gpuRef;

  h_a = (float *)malloc(nBytes);
  h_b = (float *)malloc(nBytes);

  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  initData(h_a, nx, 1.0f);
  initData(h_b, ny, 2.0f);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  float *d_a, *d_b;
  cudaMalloc((void **)&d_a, nBytes);
  cudaMalloc((void **)&d_b, nBytes);

  cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

  int dimx = 32;
  int dimy = 32;
  dim3 block(dimx, dimy);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  printf("grid:(%d,%d),Block:(%d,%d)\n",grid.x,grid.y,block.x,block.y);

  sumMatrix<<<2,32>>>(d_a);
   
  cudaMemcpy(gpuRef, d_a, nBytes, cudaMemcpyDeviceToHost);
     printf("%f\n",*gpuRef);
  //check_data(gpuRef, 10);

  cudaFree(d_a);
  cudaFree(d_b);
  free(h_a);
  free(h_b);
}
