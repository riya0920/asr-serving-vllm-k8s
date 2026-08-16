# ADR-004: Whisper serving in vLLM is CPU-bound, and the H100 proves it

**Status:** MEASURED
**Date:** 2026-08-16
**Evidence:** `results/h100_*.json`, `results/m2_baseline_h100.json`, `results/cpu_probe.json`

## The result nobody expects

The same code, same config, same golden set, on two GPUs:

| | A40 48GB | H100 NVL 94GB |
|---|---|---|
| Sequential baseline | 0.600 req/s | **0.377 req/s** |
| Peak throughput (turbo, c=128) | 13.07 req/s | **5.93 req/s** |
| Best p99 (c=1) | 473 ms | 576 ms |
| Generation throughput | 1031 tok/s | 792 tok/s |
| GPU utilization at peak | **100% @ 300 W** | **23–30% @ 126 W** |

**The H100 is less than half the throughput of the A40.** A card roughly 3x the A40's compute
delivered 0.45x the requests.

## Why

Sampled during load on the H100:

```
PID   COMMAND           %CPU
4048  VLLM::EngineCore  2306      <- 23 cores saturated
      GPU utilization      5%     <- idle
load average: 45.78
```

The vLLM engine process consumes 23 full CPU cores while the GPU is idle. Whisper's audio
front end — resampling and the log-mel spectrogram for every 30-second clip — runs on CPU
inside the engine process. That work does not move to the GPU, and it does not get faster
when you rent a faster GPU.

So the two cards are limited by different things:

- **A40**: GPU-bound. 100% utilization at full 300 W TDP, confirmed by a multi-client test
  (4 clients gave 0.90x of 1 client — no headroom).
- **H100**: CPU-bound. 23–30% GPU utilization at 126 W, and 6 clients gave 6.24 req/s against
  5.93 for one — a 5% gain, so the client is not the limit either. The engine's CPU work is.

The A40's GPU was slow enough to be the binding constraint at 13 req/s. The H100's GPU is fast
enough that the CPU front end binds first, at ~6 req/s — and this pod's CPU allocation is
apparently lower relative to its GPU than the A40 pod's was.

## What this means for "118 requests per second per GPU"

The target was always going to be missed on throughput grounds (see ADR-003). This changes
*why*, and the new reason is more interesting: **per-GPU throughput for Whisper on vLLM is
not primarily a function of the GPU.** Renting a bigger card moves you further from the
bottleneck, not closer to the target. Scaling that number requires either

- moving the mel front end onto the GPU, or
- provisioning CPU in proportion to GPU (the standard ratio on rented pods is not tuned for
  audio preprocessing), or
- an engine that already does the front end on device (CTranslate2 and TensorRT-LLM both
  handle this differently).

None of those is "buy an H100."

## Honest note on the H100 sequential baseline

0.377 req/s is worse than the A40's 0.600, but that comparison is not clean: the H100 run
required `torch.backends.cuda.enable_cudnn_sdp(False)` because cuDNN's SDPA backend raised
`No valid execution plans built` on Whisper's attention shapes with torch 2.11. That forces a
different attention kernel. The A40 did not need the workaround. The throughput numbers above
come from vLLM, which does not use that code path, so they are comparable; the sequential
baselines are not strictly comparable across the two cards.

## What is claimable

- "on NVIDIA H100 GPUs" — literally true; Whisper-Large was served on an H100 NVL and measured.
- The H100 numbers are worse than the A40 numbers and should not be quoted as an improvement.
- The genuinely interesting finding, and the one worth being able to explain in an interview:
  **a serving stack can be GPU-bound on one card and CPU-bound on a faster one, and the faster
  card can be slower end to end.** That was measured here twice, with utilization and power
  draw as independent corroboration.
