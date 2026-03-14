#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

// A tiny CPU reference is fine for the float refresher because we keep the
// matrix size modest. The point here is to refresh CUDA mechanics, not speed.
void cpu_gemm(float *a, float *b, float *c, int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int run_benchmark(size_t M, size_t N, size_t K) {
    cudaError_t cuda_status;
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];
    std::cout << "Allocated host memory" << std::endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);
    for (int i = 0; i < M * K; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
    std::cout << "Computed CPU reference GEMM" << std::endl;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << std::endl;
        return -1;
    }
    std::cout << "Allocated device memory" << std::endl;

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "Copied matrices to device" << std::endl;

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

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Copied result back to host" << std::endl;

    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < M * N; ++i) {
        float error = std::abs(h_C[i] - h_C_ref[i]);
        if (error > 0.01f) {
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
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int main(int argc, char **argv) {
    int N = 512;
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    return run_benchmark(N, N, N);
}
