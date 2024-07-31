#pragma once
#include <cuda_runtime.h>
__global__ void sumMatrix(float *a, float *b, int nx, int ny);

