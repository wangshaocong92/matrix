#include "sum_matrix.h"
#include <cstdio>
__global__ void sumMatrix(float *a, float *b, int nx, int ny) {
    int a0 = 0;
    a0++;
    __syncthreads();
    printf("%d\n",a0);
}
