#include <cuda_runtime.h>
#include <iostream>

using my_dtype = float;

/*
Level 01: CUDA refresher.

Read this first if you are rusty on CUDA syntax. Every thread computes one
output element of C. The only concepts here are:
  - `blockIdx`, `blockDim`, `threadIdx`
  - row/column indexing into a flat array
  - one bounds check
  - a simple kernel launch
*/
__global__ void kernel(my_dtype *A, my_dtype *B, my_dtype *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        my_dtype sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int BLOCK_SIZE = 32;
void matmul(my_dtype *A, my_dtype *B, my_dtype *C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE - 1)) / BLOCK_SIZE, (N + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch_float.cu"
