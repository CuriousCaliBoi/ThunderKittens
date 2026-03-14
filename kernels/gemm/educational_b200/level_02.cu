#include <cuda_bf16.h>
#include <cuda_runtime.h>

/*
Level 02: same CUDA structure, but with BF16 inputs and outputs.

This is still a plain CUDA kernel. The only new thing is that we cast BF16
products back to float for accumulation, then cast the final sum back to BF16.
That gives you a simple intuition for why lower-precision math often keeps
accumulators in higher precision.
*/
__global__ void kernel(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += __bfloat162float(A[row * N + k] * B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

int BLOCK_SIZE = 32;
void matmul(__nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE - 1)) / BLOCK_SIZE, (N + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch_bf16.cu"
