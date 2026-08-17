# ADR-011: vLLM vs faster-whisper — hypothesis refuted

**Status:** MEASURED — hypothesis rejected
**Date:** 2026-08-17
**Hardware:** Lambda A10 24GB, driver 580.105.08, CUDA 13.0
**Evidence:** `results/engineA_c*.json`, `results/engineB_c*.json`, `results/engine*_wer.json`

## The hypothesis

Every prior measurement pointed at vLLM as the constraint rather than the GPU: the A40
saturated at 13.07 req/s, the H100 was *slower* at 5.93 with the GPU at 23–30%, and moving mel
extraction to the GPU bought only 1.06x while leaving neither CPU nor GPU maxed. That pattern —
high utilization, low power, nothing saturated — reads as a serialized per-request path.

CTranslate2 is a different implementation with its own kernels and batching, so it does not
share that path. If it were several times faster on identical hardware, 118 req/s would have
become an arithmetic question of picking the right engine.

## The result

Same A10, same 50 clips, same load generator, same WER gate, 4 client processes per arm.

| engine | req/s | errors | GPU | corpus WER |
|---|---|---|---|---|
| **vLLM-Omni** | **10.72** | 0 | 100% @ 151 W | 0.0239 |
| faster-whisper (CT2) | **1.89** | 0 | 100% @ 149 W | 0.0172 |
| ratio | **0.18x** | | | |

**faster-whisper was 5.7x slower.** Both engines saturated the card — the A10's TDP is 150 W
and both sat on it. The hypothesis is refuted: vLLM is the faster engine for this workload, and
the ceiling is the model on the hardware rather than the framework.

Worth noting faster-whisper produced *better* transcripts (WER 0.0172 vs 0.0239, closer to
large-v3's 0.0160). It trades throughput for accuracy here.

## The caveat that keeps this honest

`bench/fasterwhisper_server.py` calls `BatchedInferencePipeline` **per request**, which batches
segments *within* one audio file. It does **not** batch across concurrent requests, which is
exactly what vLLM continuous batching does. So this measures naive faster-whisper serving, not
its best achievable form. A server accumulating requests into cross-request batches would score
higher — how much higher is untested.

That does not weaken the conclusion about vLLM (GPU-saturated at 10.72 req/s on this card), and
it does not rescue 118 req/s, which would need roughly 11x from a starting point of 1.89.

## Two bugs found while running it

1. **`FileNotFoundError: 'ninja'`** — vLLM shells out to the ninja binary during its compile
   step. `pip install ninja` puts it in the venv `bin`, which the engine subprocess does not
   inherit; `apt-get install ninja-build` fixed it.
2. **126 errors in 324 requests, with a clean server log.** Python's `ThreadingHTTPServer`
   defaults `request_queue_size` to 5. Under 128 concurrent clients the kernel refuses
   connections once the accept queue overflows, and those requests never reach the application —
   so it presents as the engine failing while the server log stays silent. Raising the backlog
   to 256 took errors to zero.

## Consequence

**118 req/s per GPU is closed, on four independent measurements:** A40 GPU-saturated at 13.07,
H100 CPU-heavy at 5.93, GPU-mel at 1.06x, and a competing engine at 0.18x. Scaling the A10
figure of 10.72 req/s to an H100 (roughly 3–4x) gives about 32–43 req/s.
