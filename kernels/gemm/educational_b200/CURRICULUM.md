# Educational Sequence Models + B200 Curriculum

This folder started as an educational Blackwell GEMM ladder. This curriculum
turns it into a broader training ground for two related muscles:

1. Model thinking: what computation does a sequence model need to perform?
2. Kernel thinking: how does that computation map onto real GPU hardware?

Those tracks should often evolve together, but not always. Some model lessons
need no new CUDA work yet, and some CUDA lessons are worth learning before they
cleanly map onto a full sequence model. This curriculum therefore uses three
tracks:

- Model labs: architecture, data, sample complexity, decoding behavior.
- Kernel levels: CUDA and Blackwell implementation of the core primitives.
- Systems labs: cache layout, runtime behavior, and throughput at inference.

The rule is simple:

- Sync the tracks when that reveals a useful mapping.
- Split them when one side needs focused reps on its own.

## North Star

By the end, you should be able to answer all of these from first principles:

- Why does full attention scale poorly in memory and runtime?
- Why can recurrent or state-space models learn state tracking more easily?
- Why does `O(n)` sequence complexity not automatically win on GPU?
- Why do serving systems like fixed-size state and bounded caches?
- Why might a hybrid model beat either pure attention or pure recurrence?
- How do MTP and speculative decoding change the runtime picture?

## Tracks

### Track A: Model Labs

This track studies what the model is trying to compute.

- Synthetic state-tracking tasks
- Tiny Transformers, RNNs, and SSM-like baselines
- Attention, SWA, recurrent updates, hybrid blocks
- Decoding and speculative execution

### Track B: Kernel Levels

This track studies the underlying hardware primitives.

- GEMM and epilogues
- RMSNorm and RoPE
- Naive and tiled attention
- Sliding-window attention
- Recurrent state updates and scans
- Decode-time cache/state maintenance

### Track C: Systems Labs

This track studies runtime behavior beyond an isolated kernel.

- KV-cache layout
- Fixed-state recurrent inference
- Throughput versus latency
- Draft-and-verify pipelines
- End-to-end bottlenecks for long context

## How To Use This

Each stage names a primary focus and supporting work in the other tracks.

- `Core` means the main thing to build and understand.
- `Support` means useful parallel work that should not block progress.
- `Stretch` means optional deeper work if the stage clicks.

Some stages are intentionally asymmetric. For example, attention kernels merit
multiple dedicated stages, while a model-side RNN refresher can be shorter. That
is expected and healthy.

## Stage 0: Harness

Goal: build the experimental scaffolding before chasing kernels.

### Model lab

Core:
- Implement parity, addition mod `m`, and simple state-tracking datasets.
- Support three supervision modes: `Outcome`, `CoT`, and `Aligned-CoT`.
- Build exact convergence criteria and minimal-sample search loops.

Support:
- Standardize tokenization and sequence formatting across tasks.
- Fix train/val splits and random seeds.

Deliverable:
- A tiny training harness that can answer: "How many samples did this model need
  to converge at sequence length `L`?"

### Kernel level

Core:
- Build a benchmarking harness pattern for all CUDA levels.
- Log runtime, effective bandwidth, and simple correctness metrics.

Support:
- Reuse the current `Makefile`-driven level workflow.

Deliverable:
- One shared benchmark/report template for later levels.

### Systems lab

Core:
- Define a common experiment report format.

Deliverable:
- A markdown report template with: setup, kernel config, runtime, occupancy
  notes, numerical checks, and lessons learned.

## Stage 1: Tiny Models + GEMM

Goal: connect the sample-complexity story to the matrix-multiply reality.

### Model lab

Core:
- Train a tiny Transformer and a tiny `LSTM` or `GRU` on parity and addition.
- Sweep sequence lengths and estimate minimal sample counts.

Support:
- Match parameter counts as closely as practical.
- Keep optimization recipes simple and documented.

Deliverable:
- First plots of samples needed versus sequence length.

### Kernel level

Core:
- Work through `level_01` to `level_03`.
- Understand naive GEMM, precision changes, and shared-memory tiling.

Support:
- Measure how tile size and precision change throughput.

Stretch:
- Add simple fused bias or activation epilogues after GEMM.

Deliverable:
- Notes on why both Transformers and RNNs still reduce to lots of GEMMs, but
  differ in how much parallel work they expose over sequence length.

## Stage 2: Transformer Plumbing

Goal: understand the cheap-looking building blocks around attention.

### Model lab

Core:
- Implement a clean causal Transformer block from scratch.
- Isolate embeddings, RoPE, attention, MLP, and norm.

Support:
- Verify attention masks and position handling carefully.

Deliverable:
- A minimal reference Transformer suitable for later ablations.

### Kernel level

Core:
- Add educational kernels or references for `RMSNorm` and `RoPE`.
- Study read/write traffic and reduction patterns.

Support:
- Decide which pieces are memory-bound versus compute-bound.

Deliverable:
- Short writeups for `RMSNorm` and `RoPE` as first-class sequence-model
  primitives, not just helper functions.

### Systems lab

Core:
- Introduce KV-cache concepts and memory layout choices.

Deliverable:
- Notes on why decoding changes the performance picture compared to training.

## Stage 3: Naive Attention

Goal: make full attention concrete before optimizing it.

### Model lab

Core:
- Train the tiny Transformer with explicit attention visualizations or dumps on
  synthetic tasks.

Support:
- Compare outcome-only supervision with `CoT`.

Deliverable:
- A clear explanation of what full attention buys you on these tasks.

### Kernel level

Core:
- Build a naive causal attention kernel.
- Implement `QK^T`, masking, softmax, and `PV`.

Support:
- Start with correctness and clarity, not speed.

Deliverable:
- A reference attention level that is obviously correct and obviously slow.

### Systems lab

Core:
- Measure memory footprint as sequence length grows.

Deliverable:
- A note showing exactly where attention's quadratic pain appears.

## Stage 4: Tiled / Flash-Style Attention

Goal: learn why better attention kernels are mostly about memory movement.

### Model lab

Core:
- Keep the same Transformer architecture and task setup.
- Treat this stage as mostly systems-oriented.

Support:
- Compare identical model quality under faster kernels only if convenient.

### Kernel level

Core:
- Build a tiled attention kernel with online softmax ideas.
- Study SRAM/shared-memory reuse and recomputation tradeoffs.

Support:
- Compare naive attention against tiled attention on sequence length sweeps.

Stretch:
- Add a simplified FlashAttention-style implementation note.

Deliverable:
- A report on why recomputing some values can beat storing everything.

### Systems lab

Core:
- Benchmark long-sequence latency and memory reduction.

Deliverable:
- A before/after table for naive versus tiled attention.

## Stage 5: Sliding-Window Attention + Sinks

Goal: study bounded attention as a serving-oriented compromise.

### Model lab

Core:
- Replace full attention with sliding-window attention.
- Add sink tokens or sink positions as a controlled experiment.

Support:
- Sweep window sizes such as `64`, `128`, `256`, `512`.

Deliverable:
- A plot of quality or convergence versus window size.

### Kernel level

Core:
- Implement a local attention kernel that only touches a bounded window.
- Compare cache locality and memory traffic against full attention.

Support:
- Add an explicit cache layout for bounded history.

Deliverable:
- A clean explanation of why bounded windows are infra-friendly.

### Systems lab

Core:
- Build a fixed-window decode path.
- Measure state size and latency stability over long contexts.

Deliverable:
- A runtime report on why SWA is attractive in production.

## Stage 6: Recurrent Updates

Goal: build the simplest state-update path that attention does not naturally
encourage.

### Model lab

Core:
- Train an `LSTM` or `GRU` baseline on the same tasks.
- Compare sample complexity against the Transformer baseline.

Support:
- Compare `CoT` versus `Aligned-CoT`.

Deliverable:
- First direct evidence of recurrent inductive bias on state tracking.

### Kernel level

Core:
- Implement simple recurrent state-update kernels.
- Study elementwise updates, reductions, and hidden-state movement.

Support:
- Measure the serial dependency cost across timesteps.

Deliverable:
- Notes on why recurrence often has the right bias but weaker naive parallelism.

### Systems lab

Core:
- Build a fixed-size recurrent inference state.

Deliverable:
- A comparison of recurrent state memory versus KV-cache growth.

## Stage 7: Parallelizing Recurrence

Goal: learn the hardware tricks that make recurrent computation less painful.

### Model lab

Core:
- Use this as a lighter model stage.
- Keep the model fixed and focus on how to execute recurrence better.

Deliverable:
- A short note on where recurrence remains semantically useful even when system
  complexity rises.

### Kernel level

Core:
- Implement chunked recurrence or scan-style execution for associative pieces.
- Study prefix-scan and chunk/state handoff patterns.

Support:
- Compare per-token serial execution against chunked execution.

Stretch:
- Identify what parts of the recurrence can and cannot be parallelized.

Deliverable:
- A report on the gap between algorithmic `O(n)` claims and hardware reality.

### Systems lab

Core:
- Measure latency, throughput, and occupancy tradeoffs for chunk sizes.

Deliverable:
- Guidance on when chunking helps and when it just adds overhead.

## Stage 8: Linear Attention

Goal: implement a running-state attention variant and test the promise directly.

### Model lab

Core:
- Implement a simple linear attention formulation with running state such as
  `S_t = S_{t-1} + phi(K_t)^T V_t`.
- Train it on the synthetic tasks and compare to both Transformer and RNN.

Support:
- Test whether it behaves more like attention or recurrence in practice.

Deliverable:
- A clean empirical answer to: "Did linear attention inherit the right
  inductive bias?"

### Kernel level

Core:
- Implement the state-update kernel for linear attention.
- Compare arithmetic intensity and memory movement against attention.

Support:
- Benchmark sequence length scaling.

Deliverable:
- A first-principles report on why linear attention may or may not be faster on
  real hardware.

### Systems lab

Core:
- Build decode-time fixed-state serving for the linear-attention variant.

Deliverable:
- An inference comparison: full KV cache, bounded KV cache, and running state.

## Stage 9: SSM / Mamba-Lite

Goal: study state-space sequence modeling beyond simple RNNs.

### Model lab

Core:
- Implement a simplified SSM or Mamba-like block.
- Keep the implementation intentionally pedagogical, not feature-complete.

Support:
- Compare against the linear-attention stage on the same tasks.

Deliverable:
- Notes on what extra structure SSM-style models add beyond plain recurrence.

### Kernel level

Core:
- Implement the state update plus any local mixing or convolution needed for the
  simplified block.
- Study chunk size, state layout, and fusion opportunities.

Deliverable:
- A writeup connecting "nice inductive bias" to "nontrivial execution plan."

### Systems lab

Core:
- Measure long-sequence behavior with fixed-size state.

Deliverable:
- A serving note on why these models are attractive despite more complex kernels.

## Stage 10: Hybrid Blocks

Goal: make the current hybrid-model wave tangible.

### Model lab

Core:
- Build a hybrid stack that alternates recurrent or SSM blocks with attention.
- Try simple patterns such as `3 recurrent : 1 attention`.

Support:
- Compare task quality, convergence, and robustness to sequence length changes.

Deliverable:
- A direct test of the "hybrid feels inevitable" hypothesis.

### Kernel level

Core:
- Identify which kernels must coexist in one runtime.
- Compare the operational mix of attention, state update, norm, and MLP work.

Deliverable:
- A map of the kernel inventory a hybrid model actually requires.

### Systems lab

Core:
- Build a mixed runtime that carries both bounded KV cache and recurrent state.

Deliverable:
- A note on the real systems complexity of hybrid deployment.

## Stage 11: Decode Acceleration

Goal: connect architecture to end-user throughput.

### Model lab

Core:
- Add a small draft head or MTP-style auxiliary head.
- Evaluate accept length and quality on short coding-style tasks if possible.

Support:
- Keep this stage practical rather than doctrinal.

Deliverable:
- A measured view of how architecture and decoding strategy interact.

### Kernel level

Core:
- Profile the draft and verify path.
- Identify where idle time or imbalance appears.

Deliverable:
- A runtime explanation of why MTP/speculative decoding can matter so much.

### Systems lab

Core:
- Build a simple draft-and-verify pipeline.

Deliverable:
- Throughput measurements with and without draft generation.

## Stage 12: Capstone

Goal: stop collecting parts and start making an argument.

### Model lab

Core:
- Choose one target claim to investigate:
  - Transformers learn length-specific heuristics.
  - Recurrent models share mechanisms better.
  - Sliding windows are the best practical compromise.
  - Hybrids are best overall for this workload.

Deliverable:
- A short report or post defending the claim with experiments.

### Kernel level

Core:
- Choose one kernel family to tune deeply.

Deliverable:
- A final optimized educational implementation plus a writeup of what mattered.

### Systems lab

Core:
- Build an end-to-end benchmark that exercises your chosen architecture.

Deliverable:
- A final dashboard or markdown report covering quality, runtime, memory, and
  implementation complexity.

## Mapping To Existing Levels

The current folder already covers the GEMM-heavy hardware foundation well.

- `level_01` to `level_03`: Stage 1 core kernel ladder
- `level_04` to `level_08`: advanced Blackwell lessons for the kernel track

That means the near-term extension should not replace the current GEMM work. It
should add the next educational sequence-model primitives around it:

- `norm` and position handling
- attention
- bounded attention
- recurrence and scan
- decode-time runtime machinery

## Suggested Deliverables By Folder

One possible organization:

- `kernels/gemm/educational_b200/`
  - Existing Blackwell GEMM ladder
- `kernels/sequence/educational_b200/`
  - RMSNorm, RoPE, attention, SWA, recurrence, scan
- `experiments/sequence_induction_bias/`
  - Synthetic tasks, tiny models, sample-complexity harness
- `reports/sequence_curriculum/`
  - Per-stage reports and plots

This is only a suggestion. If keeping everything under one folder is simpler for
now, do that first and split later.

## Metrics To Track Every Time

For model labs:

- Minimal samples to convergence
- Accuracy or exact-match
- Sequence-length sensitivity
- Parameter count
- Training stability notes

For kernel levels:

- Correctness against a reference
- Runtime
- Effective bandwidth
- Estimated FLOP/s when relevant
- Occupancy, register pressure, and shared-memory notes

For systems labs:

- End-to-end latency
- Tokens per second
- Steady-state memory use
- Cache or state size
- Complexity of the runtime path

## Immediate Next Steps

The first useful implementation wave is:

1. Keep using `level_01` to `level_03` as the stage-1 kernel ladder.
2. Add a small experiment harness for parity and addition tasks.
3. Add educational `RMSNorm` and `RoPE` work as the next sequence primitives.
4. Add a naive causal attention level before trying anything clever.

That gets the curriculum off the whiteboard and into code without prematurely
jumping into a full hybrid model.
