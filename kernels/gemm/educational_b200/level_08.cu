/*
Level 08: MXFP8 on Blackwell.

This is the key bridge from "I understand B200 GEMM" to "I understand why MoE
 kernels get tricky on B200." The included kernel adds scale movement to the
 pipeline:
  - FP8 inputs still come through TMA
  - scale tiles also move through the pipeline
  - scales are staged into TMEM before the MMA launches
  - the block-scaled tensor-core instruction consumes both value tiles and
    scale tiles

Suggested reading order:
  - `mxfp8_gemm::config`
  - `input_tiles_t` versus `input_scales_t`
  - producer path for tiles
  - producer path for scales
  - TMEM allocation for `A_sc_tm` and `B_sc_tm`
*/

#define main educational_b200_level_08_unused_main
#include "../mxfp8_b200/mxfp8_b200_gemm.cu"
#undef main

int main() {
    bool ncu = false;
    int N = 2048;
    run_benchmark<mxfp8_gemm::config<256, 5, 8, 12, 2, true>>(N, N, N, ncu);
    return 0;
}
