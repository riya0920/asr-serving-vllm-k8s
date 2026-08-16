# ADR-006: Multiple engines per GPU — also rejected

**Status:** MEASURED — hypothesis rejected
**Date:** 2026-08-16
**Evidence:** `results/c4_*.json`

## What was tested

ADR-005 found that after moving mel extraction to the GPU, *neither* resource was saturated:
16 of 192 CPU cores busy, GPU at 50% utilization but only ~130 W. That signature points at a
serialized per-request path inside one engine process rather than a resource ceiling.

If that diagnosis is right, the fix is not tuning one engine — it is running several. Turbo is
1.6 GB of weights against 94 GB of VRAM and KV cache peaked at 5%, so a dozen fit. Each engine
gets its own process, CUDA context and serialized path.

## Result

Four engines started sequentially (concurrent startup thrashes `torch.compile`), two clients
each, zero errors:

| configuration | throughput |
|---|---|
| 1 engine | 6.27 req/s |
| **4 engines, same physical H100** | **4.18 req/s** (1.05 per engine) |
| GPU during 4-engine run | 23–47% util, ~115 W |
| VRAM | 60.7 GB of 94 GB |

**Four engines are slower than one.** Per-engine throughput collapsed from 6.27 to 1.05.

## What it means

The engines contend rather than compose. VRAM was not the constraint (35% headroom), CPU was
not the constraint (192 cores), and the GPU was not saturated (47% at a third of TDP). What
multiple CUDA contexts on one device do add is scheduling and context-switch overhead, and for
this workload that overhead exceeds the parallelism gained.

So the serialization ADR-005 identified is not per-*process*. It sits somewhere shared —
the GPU's scheduler, the driver, or a synchronisation point in vLLM's encoder-decoder path
that every context hits. Distinguishing those needs Nsight profiling inside the engine, which
is upstream work.

## The optimization space, exhausted

Everything tried, measured, on the same golden set:

| change | effect |
|---|---|
| continuous batching (vs sequential) | **7.4x** ✅ |
| max-num-seqs 32 → 256 | **1.65x** ✅ |
| large-v3 → large-v3-turbo | **1.8x** ✅ |
| A40 → H100 | **0.45x** ✗ (ADR-004) |
| mel extraction CPU → GPU | **1.06x** ✗ (ADR-005) |
| 1 → 4 engines per GPU | **0.67x** ✗ (this ADR) |

Cumulative best: **0.600 → 13.07 req/s on A40 (21.8x)**, ~6.3 req/s on H100.

Target: 118 req/s per GPU. **Not reached, and no untried lever remains within
vLLM-Omni serving Whisper.** The three ideas that could still move it — CTranslate2,
TensorRT-LLM, FP8 — all replace the engine the resume bullet names.

## Recommendation

Stop optimizing. The measured story is strong on its own: a 21.8x improvement with every
step isolated, three bottlenecks identified by measurement rather than assumption, and two
well-designed experiments that produced clean negative results. That is a better engineering
narrative than a number that cannot be defended.
