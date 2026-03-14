# ThunderKittens Educational B200 GEMM Kernels

This folder is a Blackwell-oriented companion to `educational_h100`.

For the broader sequence-model learning plan that pairs model labs, kernel
levels, and systems labs, read `CURRICULUM.md`.

The first three levels are plain CUDA refreshers that are safe to read even if
you have not looked at CUDA in a while. The later levels switch to the real
Blackwell ideas: `tcgen05`, tensor memory (TMEM), 2-CTA MMA, persistent
scheduling, epilogue overlap, and MXFP8 scale movement.

Change the `LEVEL` field in the `Makefile` to `01` - `08`, then run:

```bash
make clean
make run
```

- Level 01: Naive float GEMM. Refresh thread/block indexing and flat memory access.
- Level 02: Naive BF16 GEMM. See how the same CUDA structure behaves with lower precision.
- Level 03: Shared-memory tiled BF16 GEMM.
- Level 04: First Blackwell tensor-core lesson. Single-CTA `tcgen05` FP8 GEMM with TMEM.
- Level 05: 2-CTA Blackwell GEMM. Learn cluster launches and CTA-pair MMA.
- Level 06: Persistent BF16 B200 GEMM. Study task scheduling and supergrouped work assignment.
- Level 07: BF16 B200 GEMM with overlap-focused configuration. Study MMA/epilogue overlap.
- Level 08: MXFP8 B200 GEMM. Study scale tiles, TMEM scale storage, and block-scaled MMA. The current standalone benchmark mirrors the upstream repo's large numerical mismatch, so use it mainly to study the pipeline.

Suggested learning order:

1. Read and run `01`, `02`, `03`.
2. Read `04` and identify the TMEM lifecycle: provision, MMA, load-back, store.
3. Read `05` and focus only on what changes when `CLUSTER_SIZE` becomes `2`.
4. Compare `06` and `07` to see how scheduling and overlap knobs change behavior.
5. Read `08` only after `04` - `07` feel comfortable.

Suggested notebook questions while reading:

- Which threads load data?
- Which warps issue tensor-core instructions?
- Where do partial sums live?
- What barrier says an input tile is ready?
- What barrier says TMEM is safe to reuse?
