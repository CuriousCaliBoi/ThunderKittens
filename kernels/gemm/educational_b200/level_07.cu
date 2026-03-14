/*
Level 07: overlap MMA and epilogue.

This level still uses the BF16 B200 kernel, but with a configuration that turns
on overlap between matrix-multiply work and epilogue work. Compare this file's
`run_benchmark` template arguments to level 06:
  - `OVERLAP_MMA_EPI`
  - `LOAD_PIPE_DEPTH`
  - `EPI_PIPE_DEPTH`
  - `NUM_CONSUMERS`

Suggested question:
  - which workers are doing compute, and which are draining TMEM to memory?
*/

#define main educational_b200_level_07_unused_main
#include "../bf16_b200/bf16_b200_gemm.cu"
#undef main

int main() {
    bool ncu = false;
    int N = 2048;
    run_benchmark<config<256, 256, 64, 8, true, 5, 4>>(N, N, N, ncu);
    return 0;
}
