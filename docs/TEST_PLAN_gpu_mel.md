# Test plan: can GPU-side mel extraction reach 118 req/s?

**Everything here is staged. The rented H100 only runs measurements.**

## The hypothesis

ADR-004 measured, on an H100 NVL serving whisper-large-v3-turbo through vLLM:

- **5.93 req/s** peak throughput
- **GPU at 5–30% utilization, 126 W** (the card's TDP is far higher)
- **`VLLM::EngineCore` at 2306% CPU** — 23 cores saturated
- 6 client processes gave 6.24 req/s vs 5.93 for one, so the client is not the limit

The engine is not waiting on the GPU. It is computing log-mel spectrograms on the CPU inside
the process that schedules GPU work.

## Why 118 is the number this predicts

Two independent routes arrive at the same place, which is the reason this is worth testing
rather than dismissing.

**From the measurement.** 5.93 req/s at 5% GPU utilization implies ~118 req/s at full
utilization. Utilization is not linear in throughput and that was a spot sample, so this is
suggestive, not proof.

**From the arithmetic.** Whisper-large's encoder is ~2.26 TFLOP per 30-second clip:

| component | per layer | × 32 layers |
|---|---|---|
| QKV projections | 7.37 GMAC | |
| attention scores + AV | 5.76 GMAC | |
| output projection | 2.46 GMAC | |
| FFN (1280 → 5120 → 1280) | 19.66 GMAC | |
| **total** | **35.2 GMAC = 70.4 GFLOP** | **2.26 TFLOP** |

At 118 req/s that is **266 TFLOP/s**. An H100 NVL sustains 400–600 TFLOP/s on large GEMMs, so
the encoder would occupy **30–45% of the card**. Turbo's decoder is negligible beside it:
~809M parameters means a decode step reads ~1.6 GB against ~3.9 TB/s of HBM3, and at batch 128
only ~73 steps/s are needed.

**Neither route says the GPU is the problem.** Both say the feeder is.

## The fix

`scripts/gpu_mel_patch.py`, installed as `sitecustomize.py` so it loads into every Python
process including the engine workers vLLM forks. It overrides
`WhisperFeatureExtractor.__call__` to pass `device="cuda"`, which switches HuggingFace from
`_np_extract_fbank_features` (numpy, CPU) to `_torch_extract_fbank_features` (torch, GPU).

Both code paths already exist upstream. Nothing currently passes anything but the default.

## Procedure

`scripts/h100_session.sh` runs it as an A/B with everything else held constant.

| arm | configuration |
|---|---|
| A | stock — mel on CPU. Must reproduce ~5.9 req/s or the box differs from the one measured |
| B | `sitecustomize.py` installed — mel on GPU |
| C | if B wins, sweep concurrency 64/128/256/512 to find the new ceiling |

Both arms use 8 client processes at concurrency 32 (256 total, matching `max-num-seqs`),
because at these rates a single Python client is itself a bottleneck. Both sample GPU
utilization **and** engine CPU together — either alone is ambiguous, the pair identifies which
resource binds.

## Success criteria

| outcome | meaning |
|---|---|
| **PASS** — B > 3x A | mel was the bottleneck; sweep for the new ceiling |
| **WEAK** — B is 1.3–3x A | bottleneck moved but something else binds; profile again |
| **FAIL** — B within 30% of A | mel was not the bottleneck; ADR-004's diagnosis is wrong |

**Any WER change invalidates the arm regardless of speed.** Changing how audio becomes
features can change transcripts. A faster wrong answer is not a result. The WER gate runs on
both arms against the same H100 baseline.

## Known risks

1. **vLLM may not use the HF feature extractor path at all** — it may have its own. The patch
   prints whether it installed; if arm B shows no CPU reduction, check that first.
2. **Downstream may expect CPU tensors.** `ASR_MEL_RETURN_CPU=1` computes on GPU and copies
   back, keeping output types identical. The script retries with it automatically if arm B
   fails to serve.
3. **A second CPU bottleneck may be hiding behind the first** — multipart parsing, resampling.
   At 118 req/s the box ingests ~113 MB/s of WAV. If arm B lands in WEAK, this is the next
   suspect, and the fix is more API server processes.
4. **vLLM's Whisper kernels may not reach 30–45% of peak** even fed perfectly. The LLM path is
   far more tuned than the encoder-decoder path.

## Cost

One H100 at ~$2.89/hr. Setup is scripted (~15 min), both arms ~20 min, sweep ~15 min. **Call
it $3–5.** Nothing else needs to be bought, and no decision waits on a human mid-session.
