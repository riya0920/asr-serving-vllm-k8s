# ASR Serving — Whisper-Large on vLLM-Omni + Kubernetes autoscaling

Elastic ASR serving: Whisper-Large-v3 served through [vLLM-Omni](https://vllm.ai/blog/vllm-omni)
with continuous batching and shared-encoder speculative decoding, on a Kubernetes fleet that
scales on queue depth via KEDA, delivered through WER-gated CI and Argo CD canary rollouts.

The design question: real ASR traffic spikes 8x at the top of the hour when meetings start.
You cannot provision for peak and idle at 12%. Can the fleet breathe — absorb the spike under
a p99 SLO, shrink back to protect cost — and can changes ship to it in minutes?

## Scoreboard

Every number here starts empty and is filled in **only** from a run artifact in `results/`.
No number in this repo comes from an estimate, a blog post, or a resume draft.

| Metric | Baseline (A40) | Measured | Target | Evidence |
|---|---|---|---|---|
| Throughput (req/s per GPU) | 0.600 | **13.07** (21.8x) | 118 ✗ unreachable | [M3](results/m3_turbo_latency.json), [ADR-003](docs/adr-003-throughput-ceiling.md) |
| Real-time factor | 18x | **392x** | — | [M3](results/m3_turbo_latency.json) |
| p99 latency, 30s clip @ c=1 | 2214 ms (max) | **473 ms** ✅ | < 620 ms | [M3](results/m3_turbo_latency.json) |
| p99 latency @ c=2 | — | 636 ms | < 620 ms | [M3](results/m3_turbo_latency.json) |
| Corpus WER, large-v3 | **0.0160** | — | gate reference | [M2](results/m2_baseline_sequential.json) |
| Corpus WER, turbo | — | **0.0236** (+0.0076) ✅ | within +0.02 | CI gate, M3 |
| Cost per audio hour | — | **−92.5%** ✅ | −55% | [M9](results/m9_cost_model.json) |
| GPU saturation proof (A40) | — | **100% util @ 300 W** | — | [ADR-003](docs/adr-003-throughput-ceiling.md) |
| Served on NVIDIA H100 | — | **yes, measured** ✅ | claim 8 | [ADR-004](docs/adr-004-cpu-bound-serving.md) |
| H100 peak throughput | — | **5.93 req/s** (worse than A40) | — | [ADR-004](docs/adr-004-cpu-bound-serving.md) |
| H100 bottleneck | — | **CPU-heavy: 23 cores busy, GPU 5%** | — | [ADR-004](docs/adr-004-cpu-bound-serving.md) |
| GPU-mel experiment | 5.90 req/s | **6.27 req/s (1.06x)** ✗ | > 3x to matter | [ADR-005](docs/adr-005-gpu-mel-negative-result.md) |
| KEDA autoscaling on queue depth | — | **2 → 16 replicas** ✅ | claim 3 | [ADR-008](docs/adr-008-keda-autoscaling-measured.md) |
| Autoscaler reaction to spike | — | **< 2 s** ✅ | claim 4 | [timeline](results/m6_wsl_spike_timeline.csv) |
| Fleet ready after spike onset | — | **28 s**, queue drained | claim 4 | [ADR-008](docs/adr-008-keda-autoscaling-measured.md) |
| p99 through 8x step spike | — | not measurable (client-bound) | < 620 ms | generator self-flagged invalid |
| Speculative decode acceptance | n/a | — | measure | 1 of 2 blockers fixed — [ADR-007](docs/adr-007-spec-decode-actual-status.md), [patch](patches/) |
| Canary promotion (analysis passes) | — | **74 s** ✅ | claim 14 | [ADR-009](docs/adr-009-canary-rollback-measured.md) |
| Auto-rollback (analysis fails) | — | **22 s** ✅ | claim 14 | [ADR-009](docs/adr-009-canary-rollback-measured.md) |
| Deploy wall-clock (commit → prod) | — | — | < 22 min | CI not yet exercised |

### Throughput and latency are a trade-off, not two independent wins

Read the first three rows together. 13.07 req/s is measured at concurrency 128, where p99 is
16 s. The 473 ms p99 is measured at concurrency 1, where throughput is 2.79 req/s. **There is
no operating point that delivers both**, because throughput comes from batching and batching
costs latency.

The architecture that reconciles them is the fleet: KEDA holds each pod at low concurrency so
p99 stays under SLO, and aggregate throughput comes from pod count. That makes high aggregate
throughput and low p99 compatible — but it makes them compatible *across the fleet*, never
*per GPU*.

**Baseline** = naive sequential Whisper-Large-v3 (fp16), one request at a time, HF transformers.
**Current** = whatever the most recent run in `results/` actually produced.

### Read the hardware line before quoting any of this

Baseline measured on an **NVIDIA A40 48GB**, not an H100 — RunPod pods cannot host Kubernetes
(see [ADR-001](docs/adr-001-substrate.md)), so the fleet work moved elsewhere and the model
work stayed here. Every *ratio* in this project is therefore honest: M3 and M4 are measured
against this same baseline on this same card. No *absolute* per-GPU throughput number here is
an H100 number, and none should be quoted as one.

No p99 is published for the baseline. 50 samples cannot measure a 99th percentile — nearest
rank would just return the maximum — so the max is reported as the max.

## Honest status

**M1 done.** RunPod standard pods cannot host a Kubernetes node: no `CAP_SYS_ADMIN`,
`ip_forward` not writable, PID 1 is not systemd. Substrate split per ADR-001.

**M2 done.** Golden set: 50 clips, each exactly 30s, whole utterances only, silence-padded,
81% speech density. Sequential baseline: 0.600 req/s, 18x real-time, WER 0.0160 — reproduced
to four decimal places on three separate hosts.

**M3 done.** Continuous batching → 256 batch slots → turbo decoder, 0.600 → 13.07 req/s
(21.8x), each step isolated. Ceiling proven to be the GPU on A40 ([ADR-003](docs/adr-003-throughput-ceiling.md))
and the CPU on H100 ([ADR-004](docs/adr-004-cpu-bound-serving.md)).

**M9 done.** Cost per audio hour −92.5% after warm headroom and idle floor.

**Artifact validation done.** `model-manifest.json` pins turbo at revision `41f01f3f…`,
11 files by SHA-256; the gate passes clean weights and fails a one-byte corruption.

**Blocked, not abandoned:** M5/M6/M8 (Kubernetes, KEDA, spike, canary, deploy timing) need a
host where root means root — every RunPod product available self-serve is a container. M4
(speculative decoding) needs implementing in vLLM first.

### A bug worth recording

The first M2 run reported WER **0.1504**. The model was fine — the golden set was not.
`build_golden.py` filled its buffer past 30s, cut the audio at exactly 30s, and dropped the
whole trailing utterance from the reference, leaving unreferenced speech in every clip.
Whisper transcribed it correctly and every word scored as an insertion.

The error breakdown is what gave it away: 45 substitutions but 277 deletions and 224
insertions. Substitutions are what a model gets *wrong*; deletions and insertions at that
ratio are an alignment problem. After the fix, substitutions stayed at 45 while deletions
fell to 5 and insertions to 7.

Had 0.1504 been committed as the CI reference, the gate would have permitted a model to
degrade to ~15% WER and still pass — the exact silent failure the gate exists to catch.

Two known risks are tracked openly rather than buried:

1. **Speculative decoding is not supported for encoder-decoder models in vLLM today**
   ([vllm#7366](https://github.com/vllm-project/vllm/issues/7366) — the enc-dec path asserts it
   off). M4 implements the narrow shared-encoder case. Until M4 lands and reports an acceptance
   rate, this project does not claim speculative decoding.
2. **Speculative decoding and continuous batching compete for the same resource.** Spec decode
   pays off because decode steps leave compute idle; continuous batching exists to fill that
   idle compute with other requests. The gain measured at batch=1 will not hold at batch=64.
   Every spec-decode number in this repo is reported at production batch size, and the batch=1
   number is reported next to it so the gap is visible.

## Layout

```
bench/       baseline harness + step-function load generator
scripts/     environment probes and setup
infra/       k8s manifests, KEDA ScaledObject, Argo CD apps  (M5+)
ci/          GitHub Actions workflows, WER gate              (M7+)
golden/      golden audio clips + reference transcripts      (M2)
results/     raw run artifacts — the only source of numbers
docs/        design notes and decision records
```
