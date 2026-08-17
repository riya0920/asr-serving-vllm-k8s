# ADR-005: Moving mel extraction to the GPU — a negative result

**Status:** MEASURED — hypothesis rejected
**Date:** 2026-08-16
**Evidence:** `results/armA_c*.json`, `results/armB_c*.json`, `results/arm*_wer.json`
**Plan:** `docs/TEST_PLAN_gpu_mel.md`

## What was tested

ADR-004 found the H100 CPU-bound: 5.9 req/s with the GPU at 5–30% and `VLLM::EngineCore`
burning 23 cores. The hypothesis was that Whisper's log-mel front end, computed on CPU inside
the engine process, was the binding constraint — and that moving it to the GPU would unlock
the throughput the GPU arithmetic says is available (2.26 TFLOP/clip × 118 req/s = 266
TFLOP/s ≈ 30–45% of an H100).

`scripts/gpu_mel_patch.py` installed as `sitecustomize.py`, flipping
`WhisperFeatureExtractor` from `_np_extract_fbank_features` (numpy/CPU) to
`_torch_extract_fbank_features` (torch/GPU). A/B on one H100 NVL, everything else identical,
8 client processes at concurrency 32 per arm.

## Result

| | ARM A (mel on CPU) | ARM B (mel on GPU) |
|---|---|---|
| Sum of per-client rps | 5.90 | **6.27** |
| Wall-clock aggregate | 5.87 | **6.24** |
| Engine CPU | 2306% (23 cores) | **~1600% (16 cores)** |
| GPU utilization | 23–30% | **48–54%** |
| GPU power | ~126 W | ~122–134 W |
| Corpus WER | 0.0239 | **0.0239** |

**The patch did exactly what it was supposed to do.** Engine CPU fell ~30%, GPU utilization
roughly doubled, and WER was byte-identical, so the work genuinely moved and the transcripts
did not change.

**Throughput improved 1.06x.** Against a pass bar of 3x, that is a rejection.

## What it means

Removing roughly 30% of the engine's CPU work bought 6% throughput. **Mel extraction was a
large CPU consumer but not the binding constraint.** ADR-004 correctly identified that the
engine was CPU-heavy; it incorrectly inferred that the CPU work was therefore what limited
throughput.

The corrected picture, after the patch:

- CPU: 16 cores still busy — resampling, multipart parsing, tokenization remain
- GPU: ~50% utilization but only ~130 W, far below the card's TDP

**Neither resource is saturated.** High utilization at low power means many small kernels with
gaps between them, not dense compute. That signature points at a *serialized critical path* —
per-request work that cannot overlap — rather than a throughput limit in either resource.
Likely candidates: the encoder running per-request instead of batched, or synchronisation
points between the audio front end and the scheduler.

## A measurement error worth recording

The session script computed arm B's aggregate as `640 / elapsed`, using the *requested* count
(8 clients × 80). The closed-loop generator overshoots — 888 requests actually completed — so
that arithmetic reported 4.48 req/s and made arm B look like a 24% regression. It was a 6%
improvement.

The error was caught by recomputing both arms from the artifacts with the same method. The
lesson is narrow and worth keeping: **never derive a rate from an expected count.** Count what
finished. Two arms compared by different methods is not a comparison at all, and the wrong
version of this result was believed for several minutes.

## Consequence for 118 req/s

Closed. Three independent findings now bound it:

1. **A40 (ADR-003)**: GPU-saturated at 13.07 req/s, 100% util at full 300 W TDP
2. **H100 (ADR-004)**: 5.9 req/s, CPU-heavy, GPU idle
3. **H100 + GPU mel (this ADR)**: CPU work reduced 30%, throughput +6%

The remaining hypothesis — a serialized per-request critical path inside vLLM's
encoder-decoder implementation — is a vLLM internals problem, not a configuration or
hardware problem. It would need profiling inside the engine and quite possibly upstream
changes, which is the same class of work as the speculative-decoding gap and equally out of
scope here.

**118 req/s per GPU is not reachable with vLLM-Omni serving Whisper.** Not for want of
hardware, and not for want of the obvious optimizations — those were tried and measured.

## What still might reach it

Untested, and each replaces part of the serving stack:

- **CTranslate2 / faster-whisper** — purpose-built for Whisper, does the front end differently
- **TensorRT-LLM** — NVIDIA's own encoder-decoder path, heavily tuned
- **FP8 on Hopper** — orthogonal, plausibly 1.5–2x on top of either

Any number earned on those is a number for that engine, not for vLLM-Omni.
