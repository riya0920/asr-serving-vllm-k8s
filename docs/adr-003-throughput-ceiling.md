# ADR-003: The measured throughput ceiling, and what it means for the 118 req/s target

**Status:** MEASURED — decision pending
**Date:** 2026-08-15
**Evidence:** `results/diagA.json`, `results/b1..b4.json`, `results/m3_*.json`

## What was measured

All on a single NVIDIA A40 48GB (driver 580.159.03, CUDA 13.0), 50 clips of exactly 30s,
`openai/whisper-large-v3-turbo` unless stated.

| configuration | req/s | vs baseline |
|---|---|---|
| HF transformers, sequential, whisper-large-v3 | 0.600 | 1.0x |
| vLLM continuous batching, large-v3, max-num-seqs 32 | 4.45 | 7.4x |
| + max-num-seqs 256 | 7.36 | 12.3x |
| + large-v3-turbo (4 decoder layers) | 13.36 | 22.3x |
| reproduced on a second A40 host | 13.07 | 21.8x |

Accuracy was gated, not assumed: turbo scored WER 0.0236 against the large-v3 baseline of
0.0160 — a real regression of +0.0076, inside the +0.02 CI tolerance.

## Is the GPU actually the limit?

This mattered because throughput flatlined near 13 req/s while the scheduler filled only ~68
of 256 slots and KV cache sat at ~5%. Both are classic signs of a *client-side* bottleneck,
and the load generator is a single Python process uploading ~960 KB per request from the same
host. If the harness were the limit, every number above would be measuring the benchmark
rather than the server.

Two tests:

| test | configuration | throughput |
|---|---|---|
| A | 1 client process, concurrency 128 | 13.07 req/s |
| B | 4 client processes, concurrency 32 each | 11.71 req/s |

**Ratio 0.90x — four clients were slightly slower, not faster.** During test B the GPU
reported **100% utilization at 300 W**, which is the A40's full rated TDP.

A card at rated power with pegged utilization is not waiting on a client. The bottleneck is
the GPU. ~13 req/s is this hardware's real ceiling for this model on this engine.

## Correction to an earlier estimate

An earlier analysis in this project estimated Whisper-large's encoder at ~2.25 TFLOP per 30s
clip, concluding that 13 req/s represented only ~30% of an A40 and that ~40 req/s was
available. **That estimate was wrong.** The saturation measurement supersedes it. Either the
FLOP model understates the true cost, or vLLM's Whisper kernels run far from peak efficiency —
distinguishing those would need profiling, but it does not change the ceiling.

Recording this because the failure mode is instructive: a plausible arithmetic argument said
there was 3x headroom, and a five-minute experiment said there was none. The experiment wins.

## Consequence for the 118 req/s target

- A40 measured ceiling: **13.07 req/s**
- H100 is ~2.5–3.5x an A40 for this class of work → **33–46 req/s**
- Target: **118 req/s**

**Roughly 3x short**, on the hardware the resume bullet names, with the engine it names, after
continuous batching, a 256-slot scheduler and the turbo decoder. Renting an H100 would not
produce 118 and should not be expected to.

## Options that could close a 3x

1. **FP8 quantization on Hopper** — plausibly 1.5–2x. Requires an H100; unavailable on Ampere.
2. **A different engine** — TensorRT-LLM or CTranslate2 (faster-whisper) are materially faster
   than vLLM for Whisper specifically. Plausibly 2–3x.
3. Both stacked could plausibly approach ~100 req/s.

The difficulty: (2) contradicts the bullet, which names vLLM-Omni. A number earned on
TensorRT-LLM cannot honestly be reported as a vLLM-Omni number.

## What is defensible today

Measured, reproduced on two hosts, accuracy-gated, artifacts committed:

- **22x throughput improvement** over a naive sequential baseline on identical hardware
- the contribution of each optimization isolated (batching 7.4x → slots 12.3x → turbo 22.3x)
- a WER gate that caught the real accuracy cost of the turbo decoder rather than hiding it
- a saturation proof establishing the ceiling is hardware, not harness

That is a stronger engineering story than an unverifiable 118, because every step of it can be
defended under questioning.
