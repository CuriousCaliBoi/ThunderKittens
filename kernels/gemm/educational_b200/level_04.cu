/*
Level 04: first Blackwell tensor-core kernel.

This level intentionally reuses the real single-CTA FP8 B200 kernel. That is
the cleanest way to meet the actual Blackwell concepts:
  1. shared-memory tiles hold A and B
  2. tensor memory (TMEM) holds the accumulator
  3. a leader warp issues `mm_ABt` / `mma_ABt`
  4. consumer warpgroups drain TMEM back to registers and then to HBM

Suggested reading order inside the included file:
  - constants and tile types
  - shared-memory allocation
  - `tensor_allocator<1, 1>`
  - producer path
  - MMA launch path
  - consumer epilogue
*/

#define main educational_b200_level_04_unused_main
#include "../fp8_b200/fp8_b200_gemm_1cta.cu"
#undef main

int main() {
    int N = 2048;
    return run_benchmark(N, N, N);
}
