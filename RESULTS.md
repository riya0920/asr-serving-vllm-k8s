# Results

Everything measured, on which hardware, with the artifact that proves it. Nothing here is
estimated. Where a number was wrong and later corrected, both versions are shown, because the
correction is usually the more interesting part.

---

## 1. The optimization chain

Single GPU, 50 clips of exactly 30s, whole utterances, silence-padded, 81% speech density.

| step | req/s | vs baseline | what changed |
|---|---|---|---|
| HF transformers, sequential | 0.600 | 1.0x | naive baseline |
| vLLM continuous batching | 4.45 | 7.4x | iteration-level scheduling |
| `--max-num-seqs 32 → 256` | 7.36 | 12.3x | KV cache was only 10.9% used |
| `whisper-large-v3-turbo` | 13.07 | **21.8x** | 4 decoder layers instead of 32 |

Reproduced on two separate A40 hosts: 13.36 and 13.07 req/s. WER 0.0160 to four decimal
places on **three** hosts, which is the strongest evidence the harness is deterministic.

**Latency, same hardware, turbo:**

| concurrency | req/s | p50 | p99 |
|---|---|---|---|
| **1** | 2.79 | 362 ms | **473 ms** ✅ under the 620 ms SLO |
| 2 | 3.90 | 525 ms | 636 ms |
| 8 | 5.48 | 1442 ms | 1902 ms |
| 128 | 13.07 | ~8500 ms | ~17000 ms |

**Throughput and latency are the same dial.** 13.07 req/s exists at concurrency 128 where p99
is 17s. The 473 ms p99 exists at concurrency 1 where throughput is 2.79 req/s. No single
operating point delivers both — a fleet reconciles them by holding each pod at low concurrency
and getting throughput from pod count, which makes them compatible *across the fleet*, never
*per GPU*.

**Cost:** −92.5% per audio hour after warm headroom and the idle-replica floor (−95.4% raw).
The ratio is price-independent, since cost per audio hour is proportional to GPU-hours per
audio-hour.

---

## 2. Where the ceiling actually is

Three experiments, each designed to fail loudly if the hypothesis was wrong.

### A40 — GPU-bound, confirmed
100% utilization at **300 W**, the card's full TDP. Four client processes delivered **0.90x**
of one — no headroom anywhere. 13.07 req/s is the hardware.

### H100 NVL — *not* GPU-bound, and slower
| | A40 | H100 NVL |
|---|---|---|
| peak throughput | 13.07 req/s | **5.93 req/s** |
| GPU utilization | 100% @ 300 W | **23–30% @ 126 W** |
| engine CPU | — | **2306% (23 cores)** |

A card with ~3x the compute delivered 0.45x the throughput. Six client processes gave 6.24 vs
5.93 — not the client either.

### The three rejected fixes

| experiment | result | verdict |
|---|---|---|
| **GPU mel extraction** ([ADR-005](docs/adr-005-gpu-mel-negative-result.md)) | engine CPU 23→16 cores, GPU util 25%→50%, throughput **1.06x**, WER identical | mel was a large CPU consumer but **not** the constraint |
| **Multiple engines per GPU** ([ADR-006](docs/adr-006-multi-instance-negative.md)) | 4 engines slower than 1 | not per-process serialization either |
| **Speculative decoding** ([ADR-007](docs/adr-007-spec-decode-actual-status.md)) | 1 of 2 blockers fixed; second is upstream | see below |

After the mel patch: **16 of 192 CPU cores busy, GPU at 50% utilization drawing only 130 W.**
Neither resource is saturated. High utilization at low power means many small kernels with
gaps between them — a serialized per-request critical path inside vLLM's encoder-decoder
implementation, which is upstream work, not configuration.

---

## 3. Speculative decoding — the corrected story

An earlier version of these notes claimed vLLM "asserts speculative decoding off for
encoder-decoder models." **That was wrong**, and it came from citing an old GitHub issue
without checking the installed source.

vLLM 0.26 **accepts** `--speculative-config` on Whisper. It builds
`SpeculativeConfig(method='draft_model')`, adjusts the scheduler, and logs *"does not fully
support multimodal models yet. Proceeding with tentative support."* Two bugs then block it:

**Blocker 1 — fixed.** `AttributeError: 'WhisperConfig' object has no attribute
'image_token_index'`. The proposer assumes every multimodal target is a *vision* model.
Whisper is multimodal by audio — no image placeholder token, no separate language submodule.
[Patch included](patches/vllm-0.26-audio-multimodal-spec-decode.patch); it fixes any audio
multimodal target and leaves every vision path untouched.

**Blocker 2 — open.** `embedding(): argument 'indices' must be Tensor, not NoneType`.
Reproduced under `--enforce-eager`, so it is runtime rather than a compile artifact: draft
input IDs are never populated on the encoder-decoder path.

Note it would not have moved throughput regardless. Speculative decoding reclaims idle compute
during memory-bound decode; batching already fills that on the A40, and the H100 is limited by
serialization. Its benefit is single-request latency at concurrency 1 — the regime already
passing at 473 ms.

---

## 4. Bugs found in this project's own measurement code

Kept because each one would have produced a confidently wrong number.

| bug | symptom | how it was caught |
|---|---|---|
| Golden set split utterances at 30s but dropped the tail from the reference | WER **0.1504** | error profile: 45 substitutions vs 277 deletions + 224 insertions. That ratio is misalignment, not a bad model. Fixed → 0.0160 |
| Percentile used `round()` — banker's rounding | p50 of 10 samples returned the 6th value | unit test |
| Rate derived from *requested* count, not completed | a 6% gain reported as a 24% loss | recomputing both arms with one method |
| Metric prefix guessed as `vllm_omni:` from HELP/TYPE lines | KEDA would query an empty series forever | counting **sample lines**: 334 `vllm:`, zero `vllm_omni:` |

Had the first one been committed as the CI reference, the gate would have permitted a model to
degrade to ~15% WER and still pass — precisely the silent failure it exists to catch.

---

## 5. Environment traps

Each cost real time and none is in any tutorial.

1. **RunPod containers cannot host Kubernetes** — no `CAP_SYS_ADMIN`, `ip_forward` unwritable,
   PID 1 is not systemd. Verified on two different templates. Bare Metal is sales-gated;
   Instant Clusters are explicitly not Kubernetes-compatible.
2. **The CUDA 13 split** — as of Aug 2026 the default PyPI wheels for both torch *and* vLLM
   require CUDA 13, while many rentable datacenter GPUs run 12.x drivers a tenant cannot
   upgrade. torch has a pinnable cu128 index; vLLM does not.
3. **vllm-omni pins vLLM's minor version.** `pip install vllm vllm-omni` yields a broken pair
   that fails with `ImportError: MistralToolCall` — an error in a *tool parser*, nothing to do
   with audio.
4. **torchcodec/torch ABI mismatch** — vLLM eagerly imports torchcodec for video; its latest
   release does not load against torch 2.11. Stub included.
5. **PEP 668** — Ubuntu 24.04 refuses pip installs into system Python.

---

## 6. Claim scorecard

**7 of 15 met.**

| # | claim | status |
|---|---|---|
| 1 | Whisper-Large via vLLM-Omni | ✅ |
| 2 | on a Kubernetes cluster | ❌ needs root |
| 3 | KEDA autoscaling, GPU util + queue depth | ❌ written, validated, never run |
| 4 | absorbs 8x spikes | ❌ needs a cluster |
| 5 | p99 < 620 ms | ✅ **473 ms** |
| 6 | continuous batching | ✅ **7.4x** |
| 7 | speculative decoding | ❌ 1 of 2 blockers fixed |
| 8 | on NVIDIA H100 | ✅ |
| 9 | 42 → 118 req/s per GPU | ❌ **13.07** measured |
| 10 | cost per audio hour −55% | ✅ **−92.5%** |
| 11 | load tests in pipeline | ✅ |
| 12 | model artifact validation | ✅ |
| 13 | GitHub Actions pipeline | ❌ no remote |
| 14 | Argo CD canary | ❌ needs a cluster |
| 15 | deploy 6 h → 22 min | ❌ needs both |

**Six of the eight gaps need one Linux host with real root** — Oracle free tier, WSL2, or any
VPS. Two are genuinely hard: #7 needs upstream vLLM work, #9 is not reachable on this stack.

### On 42 → 118

The bullet claims 2.8x. This project measured **21.8x**. The *ratio* is far better than
claimed; the *absolute numbers* are far lower. A naive Whisper-Large baseline is 0.6 req/s on
A40 and 0.377 on H100 — nothing resembling 42. For 42 to be a naive baseline, something
fundamental would have to differ: much shorter clips, many GPUs counted as one, or audio-
seconds counted as requests.
