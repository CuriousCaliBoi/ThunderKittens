/*
Level 05: CTA-pair GEMM on Blackwell.

This level switches from a single CTA to a 2-CTA cluster. Focus on what changes
relative to level 04:
  - the kernel has `__cluster_dims__(2)`
  - the CTA rank matters
  - the leader CTA issues `mm2_ABt` / `mma2_ABt`
  - B tiles are shared across the CTA pair through DSMEM
  - completion uses a cluster-aware arrival/commit path

Suggested question:
  - what work is duplicated per CTA, and what work is shared by the pair?
*/

#define main educational_b200_level_05_unused_main
#include "../fp8_b200/fp8_b200_gemm_2cta.cu"
#undef main

int main() {
    int N = 4096;
    return run_benchmark(N, N, N);
}
