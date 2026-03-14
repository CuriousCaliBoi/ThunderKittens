/*
Level 06: persistent BF16 B200 GEMM.

This level uses the full BF16 Blackwell kernel, but runs just one educational
configuration. Read it to understand:
  - supergroup scheduling via `get_swizzled_2d_idx`
  - persistent task scheduling with `clc::schedule`
  - TMEM provisioning and deprovisioning
  - producer warps versus epilogue warps

Suggested question:
  - where does the next tile get chosen, and how does that reduce tail effects?
*/

#define main educational_b200_level_06_unused_main
#include "../bf16_b200/bf16_b200_gemm.cu"
#undef main

int main() {
    bool ncu = false;
    int N = 4096;
    run_benchmark<config<256, 256, 64, 4, false, 4, 8>>(N, N, N, ncu);
    return 0;
}
