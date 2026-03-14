#include <chrono>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#include "../common.cuh"

// This harness matches the original educational flow but uses a smaller default
// size so the refresher levels finish quickly on a rental machine.
int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cuda_status;
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    __nv_bfloat16 *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&d_B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C, M * N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_C_ref, M * N * sizeof(__nv_bfloat16));
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[M * K];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[K * N];
    for (int i = 0; i < M * K; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    cudaMemcpy(d_A, h_A_bf16, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_bf16, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

    reference_gemm<__nv_bfloat16, __nv_bfloat16, false>(d_C_ref, d_A, d_B, M, N, K);
    cudaDeviceSynchronize();
    std::cout << "Computed reference GEMM on device" << std::endl;

    for (int i = 0; i < 2; i++) matmul(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    constexpr int ITERS = 5;
    for (int i = 0; i < ITERS; i++) {
        matmul(d_A, d_B, d_C, M);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    double useconds = diff.count() * 1e6 / ITERS;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / useconds) / 1e6;
    std::cout << "Avg kernel execution time: " << useconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }

    __nv_bfloat16 *h_C_bf16 = new __nv_bfloat16[M * N];
    __nv_bfloat16 *h_C_ref_bf16 = new __nv_bfloat16[M * N];
    cudaMemcpy(h_C_bf16, d_C, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref_bf16, d_C_ref, M * N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    std::cout << "Copied result back to host" << std::endl;

    float *h_C_ref = new float[M * N];
    for (int i = 0; i < M * N; ++i) h_C[i] = __bfloat162float(h_C_bf16[i]);
    for (int i = 0; i < M * N; ++i) h_C_ref[i] = __bfloat162float(h_C_ref_bf16[i]);

    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if (error > 0.2f) {
            if (error_count < 20) {
                std::cout << "Error at row " << i / N << " col " << i % N
                          << ": " << h_C[i] << " != " << h_C_ref[i] << " (ref)" << std::endl;
            } else if (error_count == 21) {
                std::cout << "Too many errors to show them all.\n";
            }
            error_count++;
        }
        max_error = std::max(max_error, error);
    }

    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Error count: " << error_count << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_C_bf16;
    delete[] h_C_ref_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);

    return 0;
}

int main(int argc, char **argv) {
    int N = 1024;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    return run_benchmark(N, N, N);
}
